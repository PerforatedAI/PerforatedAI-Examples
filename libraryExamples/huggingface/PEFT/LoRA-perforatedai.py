# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import argparse
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    tuners,
)

import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm

from perforatedai import pb_globals as PBG
from perforatedai import pb_models as PBM
from perforatedai import pb_utils as PBU
import sys

saveName = 'PBgpt2'

model_name_or_path = "gpt2-medium"
PBG.fixedSwitchNum = 50
PBG.firstFixedSwitchNum = 49
# One Dendrite works best with our example code, may not be best for your dataset.
PBG.maxDendrites = 1
#last working one had num_epochs 50
#50 has a better starting rate, but still need to run with doing history
num_epochs = 50
    
batch_size = 32
task = "mrpc"
peft_type = PeftType.LORA
device = "cuda"


# When to switch between Dendrite learning and neuron learning. 
PBG.switchMode = PBG.doingHistory 
# How many normal epochs to wait for before switching modes, make sure this is higher than your scheduler's patience.
PBG.nEpochsToSwitch = 10  
# Same as above for Dendrite epochs
PBG.pEpochsToSwitch = 10
# The default shape of input tensors
PBG.inputDimensions = [-1, -1, 0]






peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
lr = 3e-4

# +
if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

datasets = load_dataset("glue", task)
metricVal = evaluate.load("glue", task)
metricTest = evaluate.load("glue", task)


def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
    return outputs


tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "sentence1", "sentence2"],
)

# We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
# transformers library
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")


def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


# Instantiate dataloaders.
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
)
test_dataloader = DataLoader(
    tokenized_datasets["test"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
)
# -

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model.config.pad_token_id = tokenizer.eos_token_id
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model
#clear the original to not convert non-lora modules
PBG.modulesToConvert = [tuners.lora.layer.Linear]
PBG.unwrappedModulesConfirmed = True
PBG.unwrappedNormsConfirmed = True
PBG.usingSafeTensors = False

PBG.pbImprovementThreshold = 0.5 #improvement increase needed to call a new best PBScore
PBG.pbImprovementThresholdRaw = 1e-4# raw increase needed, if its lower than this its not really learning 


model = PBU.convertNetwork(model)
PBG.pbTracker.initialize(
    doingPB = True, #This can be set to false if you want to do just normal training 
    saveName=saveName,  # Change the save name for different parameter runs
    maximizingScore=True, # True for maximizing validation score, false for minimizing validation loss
    makingGraphs=True)  # True if you want graphs to be saved


# +
optimizer = AdamW(params=model.parameters(), lr=lr)

# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
    num_training_steps=(len(train_dataloader) * num_epochs),
)

PBG.pbTracker.setOptimizerInstance(optimizer)
# -

model.to(device)
epoch = 0
while(True):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metricVal.add_batch(
            predictions=predictions,
            references=references,
        )
    eval_metric = metricVal.compute()
    for step, batch in enumerate(tqdm(test_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metricTest.add_batch(
            predictions=predictions,
            references=references,
        )
    test_metric = metricTest.compute()
    print(f"epoch eval {epoch}:", eval_metric)
    print(f"epoch test {epoch}:", test_metric)
    PBG.pbTracker.addTestScore(test_metric['accuracy'], 'Test Accuracy')
    model, improved, restructured, trainingComplete = PBG.pbTracker.addValidationScore(eval_metric['accuracy'], 
    model, # .module if its a dataParallel
    saveName)
    model.to(device)
    if(trainingComplete):
        break
    elif(restructured):
        if(PBG.pbTracker.memberVars['mode'] == 'n'):
            optimizer = AdamW(params=model.parameters(), lr=(lr/2))
        else:
            optimizer = AdamW(params=model.parameters(), lr=(lr))
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
            num_training_steps=(len(train_dataloader) * num_epochs),
        )
        PBG.pbTracker.setOptimizerInstance(optimizer)
    epoch += 1


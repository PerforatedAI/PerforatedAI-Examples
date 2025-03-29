"""
Patches the Trainer class to support arbitrary eval metrics.

Make sure you are using the PerforatedAI version of the Transformers library. Install it with:

    git clone https://github.com/PerforatedAI/PerforatedAI-Transformers.git
    cd PerforatedAI-Transformers
    pip install -e .
"""

from transformers import Trainer
from transformers.trainer_utils import SaveStrategy
from transformers.trainer_callback import CallbackHandler
import torch
from typing import Dict
from perforatedai import pb_globals as PBG

def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time):
    if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
        # if is_torch_xla_available():
        #     xm.mark_step()

        logs: Dict[str, float] = {}

        # all_gather + mean() to get average loss over all processes
        tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

        # reset tr_loss to zero
        tr_loss -= tr_loss

        logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
        if grad_norm is not None:
            logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        logs["learning_rate"] = self._get_learning_rate()

        self._total_loss_scalar += tr_loss_scalar
        self._globalstep_last_logged = self.state.global_step
        self.store_flos()

        self.log(logs, start_time)

    metrics = None
    trainingComplete = False
    if self.control.should_evaluate:
        metrics = self._evaluate(trial, ignore_keys_for_eval)
        ## Original PB Trainer code:
        # self.model, improved, restructured, trainingComplete = PBG.pbTracker.addValidationScore(metrics['eval_loss'], model, PBG.saveName)
        
        ## Patch:
        ## Use a configurable metric key from PBG; default to 'eval_loss'
        pb_metric = getattr(PBG, 'metric', 'eval_loss')
        score = metrics.get(pb_metric, metrics.get('eval_loss'))
        print(f"Using metric for PB: {pb_metric}, score: {score}")
        self.model, improved, restructured, trainingComplete = PBG.pbTracker.addValidationScore(score, model, PBG.saveName)

        model = self.model
        if(restructured):
            self.optimizer = None
            self.lr_scheduler = None
            self.create_optimizer_and_scheduler(num_training_steps=self.max_steps)
            self.callback_handler.optimizer = self.optimizer
            self.callback_handler = CallbackHandler(
                self.callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
            )
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
            
                
            PBG.pbTracker.setOptimizerInstance(self.optimizer)
        self.model_wrapped = self.model
        self.callback_handler.model = self.model
        self._move_model_to_device(self.model, self.args.device)            
        

        is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

        if self.args.save_strategy == SaveStrategy.BEST:
            self.control.should_save = is_new_best_metric

    '''
    if self.control.should_save:
        self._save_checkpoint(model, trial)
        self.control = self.callback_handler.on_save(self.args, self.state, self.control)
    '''
    return model, trainingComplete


Trainer._maybe_log_save_evaluate = _maybe_log_save_evaluate

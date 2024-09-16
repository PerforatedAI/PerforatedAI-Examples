#pylint: disable=E1101, E0401, E1102, W0621, W0221
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time

from random import SystemRandom
import modelsPAI as models
import utils

from perforatedai import globalsFile as gf
from perforatedai import pb_models as PBM
from perforatedai import pb_utils as PBU

parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=200000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--std', type=float, default=0.01)
parser.add_argument('--latent-dim', type=int, default=32)
parser.add_argument('--rec-hidden', type=int, default=32)
parser.add_argument('--gen-hidden', type=int, default=50)
parser.add_argument('--embed-time', type=int, default=128)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--enc', type=str, default='mtan_rnn')
parser.add_argument('--dec', type=str, default='mtan_rnn')
parser.add_argument('--fname', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--n', type=int, default=8000)
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--quantization', type=float, default=0.1, 
                    help="Quantization on the physionet dataset.")
parser.add_argument('--classif', action='store_true', 
                    help="Include binary classification loss")
parser.add_argument('--freq', type=float, default=10.)
parser.add_argument('--k-iwae', type=int, default=10)
parser.add_argument('--norm', action='store_true')
parser.add_argument('--kl', action='store_true')
parser.add_argument('--learn-emb', action='store_true')
parser.add_argument('--dataset', type=str, default='physionet')
parser.add_argument('--alpha', type=int, default=100.)
parser.add_argument('--old-split', type=int, default=1)
parser.add_argument('--nonormalize', action='store_true')
parser.add_argument('--enc-num-heads', type=int, default=1)
parser.add_argument('--dec-num-heads', type=int, default=1)
parser.add_argument('--num-ref-points', type=int, default=128)
parser.add_argument('--classify-pertp', action='store_true')
parser.add_argument('--multiplier', type=float, default=1)

args = parser.parse_args()


gf.paramValsSetting = gf.paramValsByUpdateEpoch

gf.switchMode = gf.doingFixedSwitch
gf.fixedSwitchNum = 200
gf.firstFixedSwitchNum = 199

gf.nodeIndex = 2
gf.inputDimensions = [-1, -1, 0]
gf.initialCorrelationBatches = 25
gf.nEpochsToSwitch = 50
gf.pEpochsToSwitch = 50
gf.capAtN = False #Makes sure subsequent rounds last max as long as first round

gf.modulesToConvert.append(models.multiTimeAttention)

class GRUCellProcessor():
    #Post processing does eventually need to return h_t and c__t, but h_t gets modified py the PB
    #nodes first so it needs to be extracted in post 1, and then gets added back in post 2
    def post_n1(self, *args, **kawrgs):
        h_t = args[0][0]
        c_t = args[0][1]
        #store the cell state temporarily and just use the hidden state to do PB functions
        self.c_t_n = c_t
        return h_t
    def post_n2(self, *args, **kawrgs):
        h_t = args[0]
        return h_t, self.c_t_n
    #these Grus are just getting passed input and no hidden state for some reason so just pass it along
    def pre_d(self, *args, **kwargs):
        return args, kwargs
        
    #for post processsing its just getting passed the output, which is (h_t,c_t). Then it wants to just pass along h_t as the output for the function to be passed to the parent while retaining both
    def post_d(self, *args, **kawrgs):
        h_t = args[0][0]
        c_t = args[0][1]
        self.h_t_d = h_t
        self.c_t_d = c_t
        return h_t
    
#the classifier gru is passing along the cell state instead of the hidden state so use that isntead
class ReverseGRUCellProcessor():
    #Post processing does eventually need to return h_t and c__t, but c_t gets modified py the PB
    #nodes first so it needs to be extracted in post 1, and then gets added back in post 2
    def post_n1(self, *args, **kawrgs):
        h_t = args[0][0]
        c_t = args[0][1]
        #store the cell state temporarily and just use the hidden state to do PB functions
        self.h_t_n = h_t
        return c_t
    def post_n2(self, *args, **kawrgs):
        c_t = args[0]
        return self.h_t_n, c_t
    #these Grus are just getting passed input and no hidden state for some reason so just pass it along
    def pre_d(self, *args, **kwargs):
        return args, kwargs
        
    #for post processsing its just getting passed the output, which is (h_t,c_t). Then it wants to just pass along h_t as the output for the function to be passed to the parent while retaining both
    def post_d(self, *args, **kawrgs):
        h_t = args[0][0]
        c_t = args[0][1]
        self.h_t_d = h_t
        self.c_t_d = c_t
        return c_t

gf.modulesToConvert.append(models.mtan_time_embedder)

gf.modulesToConvert.append(nn.GRU)
gf.modluesWithProcessing.append(nn.GRU)
gf.moduleProcessingClasses.append(GRUCellProcessor)
gf.modulesToConvert.append(models.reverseGru)
gf.modluesWithProcessing.append(models.reverseGru)
gf.moduleProcessingClasses.append(ReverseGRUCellProcessor)

#Define a full model so it all gets converted
class fullModel(nn.Module):
    def __init__(self, rec, dec, classifier):
        super(fullModel, self).__init__()
        self.rec = rec
        self.dec = dec
        self.classifier = classifier
        
    def forward(self, observed_data, observed_mask, observed_tp):
        out = self.rec(torch.cat((observed_data, observed_mask), 2), observed_tp)
        qz0_mean, qz0_logvar = out[:, :, :args.latent_dim], out[:, :, args.latent_dim:]
        epsilon = torch.randn(args.k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
        pred_y = self.classifier(z0)
        pred_x = self.dec(
            z0, observed_tp[None, :, :].repeat(args.k_iwae, 1, 1).view(-1, observed_tp.shape[1]))
        pred_x = pred_x.view(args.k_iwae, batch_len, pred_x.shape[1], pred_x.shape[2]) #nsample, batch, seqlen, dim
        return pred_x, pred_y, qz0_mean, qz0_logvar


if __name__ == '__main__':
    experiment_id = int(SystemRandom().random()*100000)
    print(args, experiment_id)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.dataset == 'physionet':
        data_obj = utils.get_physionet_data(args, 'cpu', args.quantization)
    elif args.dataset == 'mimiciii':
        data_obj = utils.get_mimiciii_data(args)
    
    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    val_loader = data_obj["val_dataloader"]
    dim = data_obj["input_dim"]
    
    internal = int(100 * args.multiplier)
    nlin = int(50 * args.multiplier)
    embed_time = 128
    embed_time = int(embed_time * args.multiplier)
    args.latent_dim = int(args.latent_dim * args.multiplier)
    args.rec_hidden = int(args.rec_hidden * args.multiplier)
    args.gen_hidden = int(args.gen_hidden * args.multiplier)
        
    
    if args.enc == 'enc_rnn3':
        rec = models.enc_rnn3(
            dim, torch.linspace(0, 1., 128), args.latent_dim, args.rec_hidden, nlin, 128 , learn_emb=args.learn_emb).to(device)
    elif args.enc == 'mtan_rnn':
        rec = models.enc_mtan_rnn(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.rec_hidden, 
            embed_time=embed_time, learn_emb=args.learn_emb, num_heads=args.enc_num_heads, device=device).to(device)

    if args.dec == 'rnn3':
        dec = models.dec_rnn3(
            dim, torch.linspace(0, 1., 128), args.latent_dim, args.gen_hidden, nlin, 128, learn_emb=args.learn_emb).to(device)
    elif args.dec == 'mtan_rnn':
        dec = models.dec_mtan_rnn(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.gen_hidden, 
            embed_time=embed_time, learn_emb=args.learn_emb, num_heads=args.dec_num_heads, device=device).to(device)
    
    classifier = models.create_classifier(args.latent_dim, internal, args.rec_hidden).to(device)
    
    model = fullModel(rec, dec, classifier)
    model = PBU.convertNetwork(model)

    gf.pbTracker.initialize(doingPB = True, saveName='mTANPB', maximizingScore=True, makingGraphs=True)

    model.classifier.classifier[0].setThisInputDimensions([-1, 0])
    model.classifier.classifier[2].setThisInputDimensions([-1, 0])
    model.classifier.classifier[4].setThisInputDimensions([-1, 0])
    
    gf.pbTracker.setOptimizer(optim.Adam)
    optimArgs = {'params':model.parameters(),'lr':args.lr}
    
    gf.pbTracker.setScheduler(torch.optim.lr_scheduler.MultiStepLR)
    schedArgs = {'milestones':[50,100,150], 'gamma':0.2}

    optimizer, scheduler = gf.pbTracker.setupOptimizer(model, optimArgs, schedArgs)
    criterion = nn.CrossEntropyLoss()
    
    if args.fname is not None:
        checkpoint = torch.load(args.fname)
        rec.load_state_dict(checkpoint['rec_state_dict'])
        dec.load_state_dict(checkpoint['dec_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading saved weights', checkpoint['epoch'])

    best_val_loss = float('inf')
    total_time = 0.
    for itr in range(1, args.niters + 1):
        train_recon_loss, train_ce_loss = 0, 0
        mse = 0
        train_n = 0
        train_acc = 0
        #avg_reconst, avg_kl, mse = 0, 0, 0
        if args.kl:
            wait_until_kl_inc = 10
            if itr < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1-0.99** (itr - wait_until_kl_inc))
        else:
            kl_coef = 1
        start_time = time.time()
        for train_batch, label in train_loader:
            train_batch, label = train_batch.to(device), label.to(device)
            batch_len  = train_batch.shape[0]
            observed_data, observed_mask, observed_tp \
                = train_batch[:, :, :dim], train_batch[:, :, dim:2*dim], train_batch[:, :, -1]

            pred_x, pred_y, qz0_mean, qz0_logvar = model(observed_data, observed_mask, observed_tp)
            # compute loss
            logpx, analytic_kl = utils.compute_losses(
                dim, train_batch, qz0_mean, qz0_logvar, pred_x, args, device)
            recon_loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean(0) - np.log(args.k_iwae))
            label = label.unsqueeze(0).repeat_interleave(args.k_iwae, 0).view(-1)
            ce_loss = criterion(pred_y, label)
            loss = recon_loss + args.alpha*ce_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_ce_loss += ce_loss.item() * batch_len
            train_recon_loss += recon_loss.item() * batch_len
            train_acc += (pred_y.argmax(1) == label).sum().item()/args.k_iwae
            train_n += batch_len
            mse += (utils.mean_squared_error(observed_data, pred_x.mean(0), 
                                      observed_mask) * batch_len).item()
        total_time += time.time() - start_time
        model.eval()
        val_loss, val_acc, val_auc = utils.evaluate_classifier(
            model.rec, val_loader, args=args, classifier=model.classifier, reconst=True, num_sample=1, dim=dim, device=device)
        model.train()

        if val_loss <= best_val_loss:
            best_val_loss = min(best_val_loss, val_loss)
            rec_state_dict = model.rec.state_dict()
            dec_state_dict = model.dec.state_dict()
            classifier_state_dict = model.classifier.state_dict()
            optimizer_state_dict = optimizer.state_dict()
        model.eval()
        test_loss, test_acc, test_auc = utils.evaluate_classifier(
            model.rec, test_loader, args=args, classifier=model.classifier, reconst=True, num_sample=1, dim=dim, device=device)
        model.train()
        print('Iter: {}, recon_loss: {:.4f}, ce_loss: {:.4f}, acc: {:.4f}, mse: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, test_acc: {:.4f}, test_auc: {:.4f}'
              .format(itr, train_recon_loss/train_n, train_ce_loss/train_n, 
                      train_acc/train_n, mse/train_n, val_loss, val_acc, test_acc, test_auc))

        train_acc = train_acc/train_n
        gf.pbTracker.addExtraScore(train_acc, 'train_acc')
        gf.pbTracker.addExtraScore(val_acc, 'val_acc')
        gf.pbTracker.addTestScore(test_auc, 'test_auc')
        gf.pbTracker.addExtraScore(test_acc, 'test_acc')
        
        model, improved, restructured, trainingComplete = gf.pbTracker.addValidationScore(val_auc, model, 'mTANPB')
        model.to(device)
        if(trainingComplete):
            break
        if(restructured):
            optimArgs = {'params':model.parameters(),'lr':args.lr}
            schedArgs = {'milestones':[50,100,150], 'gamma':0.2}

            optimizer, scheduler = gf.pbTracker.setupOptimizer(model, optimArgs, schedArgs)

            
        if itr % 100 == 0 and args.save:
            torch.save({
                'args': args,
                'epoch': itr,
                'rec_state_dict': rec_state_dict,
                'dec_state_dict': dec_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'classifier_state_dict': classifier_state_dict,
                'loss': -loss,
            }, args.dataset + '_' + 
                args.enc + '_' + 
                args.dec + '_' + 
                str(experiment_id) +
                '.h5')
    print(best_val_loss)
    print(total_time)

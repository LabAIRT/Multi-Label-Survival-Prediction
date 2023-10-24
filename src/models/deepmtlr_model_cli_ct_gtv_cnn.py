from typing import Any, List
import pandas as pd

import torch
from torch import nn
from torchmtlr import MTLR

import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from torch.optim import Adam, AdamW, SGD, Adamax
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, CyclicLR

import torch.nn as nn

from torchmtlr import mtlr_neg_log_likelihood, mtlr_survival, mtlr_risk
import numpy as np
from scipy.spatial import cKDTree

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold

from src.models.components.net_cnn import Dual_MTLR


class DEEP_MTLR(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
       
        **kwargs
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()


        if self.hparams['model'] == 'Dual':
            self.model = Dual_MTLR(hparams = self.hparams)

        else:
            print('Please select the correct model architecture name.')

        self.apply(self.init_params)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def init_params(self, m: torch.nn.Module):
        """Initialize the parameters of a module.
        Parameters
        ----------
        m
            The module to initialize.
        Notes
        -----
        Convolutional layer weights are initialized from a normal distribution
        as described in [1]_ in `fan_in` mode. The final layer bias is
        initialized so that the expected predicted probability accounts for
        the class imbalance at initialization.
        References
        ----------
        .. [1] K. He et al. ‘Delving Deep into Rectifiers: Surpassing
           Human-Level Performance on ImageNet Classification’,
           arXiv:1502.01852 [cs], Feb. 2015.
        """

        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, a=.1)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            # initialize the final bias so that the predictied probability at
            # init is equal to the proportion of positive samples
            nn.init.constant_(m.bias, -1.5214691)



    def step(self, batch: Any):
        (sample, clin_var), (y1,y2,y3,y4), labels = batch
        logits1, logits2,logits3, logits4 = self.forward((sample,clin_var))
        # print('pred_mask shape',pred_mask.shape)
        

        loss_mtlr = mtlr_neg_log_likelihood(logits1, y1.float(), self.model, self.hparams['C1'], average=True)
        loss_mtlr += mtlr_neg_log_likelihood(logits2, y2.float(), self.model, self.hparams['C1'], average=True)
        loss_mtlr += mtlr_neg_log_likelihood(logits3, y3.float(), self.model, self.hparams['C1'], average=True)
        loss_mtlr += mtlr_neg_log_likelihood(logits4, y4.float(), self.model, self.hparams['C1'], average=True)
        # print('loss_mtlr', loss_mtlr)

        loss = loss_mtlr
        

        return loss, logits1, logits2,logits3, logits4, y1,y2,y3,y4, labels

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds1, preds2, preds3, preds4, y1,y2,y3,y4, labels = self.step(batch)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        torch.cuda.empty_cache()

        return {"loss": loss, "preds1": preds1, "preds2": preds2, "preds3": preds3, "preds4": preds4,"labels": labels}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        loss        = torch.stack([x["loss"] for x in outputs]).mean()

        pred_prob1   = torch.cat([x["preds1"] for x in outputs]).cpu() 
        true_time1   = torch.cat([x["labels"]["time1"] for x in outputs]).cpu()
        true_event1  = torch.cat([x["labels"]["event1"] for x in outputs]).cpu()
        pred_risk1 = mtlr_risk(pred_prob1).detach().numpy()  
        ci_event1  = concordance_index(true_time1, -pred_risk1, event_observed=true_event1)
        
        pred_prob2   = torch.cat([x["preds2"] for x in outputs]).cpu() 
        true_time2   = torch.cat([x["labels"]["time2"] for x in outputs]).cpu()
        true_event2  = torch.cat([x["labels"]["event2"] for x in outputs]).cpu()
        pred_risk2 = mtlr_risk(pred_prob2).detach().numpy()  
        ci_event2  = concordance_index(true_time2, -pred_risk2, event_observed=true_event2)
        
        pred_prob3   = torch.cat([x["preds3"] for x in outputs]).cpu() 
        true_time3   = torch.cat([x["labels"]["time3"] for x in outputs]).cpu()
        true_event3  = torch.cat([x["labels"]["event3"] for x in outputs]).cpu()
        pred_risk3 = mtlr_risk(pred_prob3).detach().numpy()  
        ci_event3  = concordance_index(true_time3, -pred_risk3, event_observed=true_event3)
        
        pred_prob4   = torch.cat([x["preds4"] for x in outputs]).cpu() 
        true_time4   = torch.cat([x["labels"]["time4"] for x in outputs]).cpu()
        true_event4  = torch.cat([x["labels"]["event4"] for x in outputs]).cpu()
        pred_risk4 = mtlr_risk(pred_prob4).detach().numpy()  
        ci_event4  = concordance_index(true_time4, -pred_risk4, event_observed=true_event4)

        print('********************************************************')

        print('train loss is ',loss)
        print('train OS ci_event is ',ci_event1)
        print('train LFFS ci_event is ',ci_event2)
        print('train RFFS ci_event is ',ci_event3)
        print('train DFFS ci_event is ',ci_event4)

        log = {"train/OS_CI": ci_event1,
               "train/LFFS_CI": ci_event2,
               "train/RFFS_CI": ci_event3,
               "train/DFFS_CI": ci_event4,
               }  
        self.log_dict(log)

        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds1, preds2, preds3, preds4, y1,y2,y3,y4, labels = self.step(batch)
        return {"loss": loss, "preds1": preds1, "preds2": preds2, "preds3": preds3, "preds4": preds4,"labels": labels}

    def validation_epoch_end(self, outputs: List[Any]):
        loss        = torch.stack([x["loss"] for x in outputs]).mean()

        pred_prob1   = torch.cat([x["preds1"] for x in outputs]).cpu() 
        true_time1   = torch.cat([x["labels"]["time1"] for x in outputs]).cpu()
        true_event1  = torch.cat([x["labels"]["event1"] for x in outputs]).cpu()
        pred_risk1 = mtlr_risk(pred_prob1).detach().numpy()  
        ci_event1  = concordance_index(true_time1, -pred_risk1, event_observed=true_event1)
        
        pred_prob2   = torch.cat([x["preds2"] for x in outputs]).cpu() 
        true_time2   = torch.cat([x["labels"]["time2"] for x in outputs]).cpu()
        true_event2  = torch.cat([x["labels"]["event2"] for x in outputs]).cpu()
        pred_risk2 = mtlr_risk(pred_prob2).detach().numpy()  
        ci_event2  = concordance_index(true_time2, -pred_risk2, event_observed=true_event2)
        
        pred_prob3   = torch.cat([x["preds3"] for x in outputs]).cpu() 
        true_time3   = torch.cat([x["labels"]["time3"] for x in outputs]).cpu()
        true_event3  = torch.cat([x["labels"]["event3"] for x in outputs]).cpu()
        pred_risk3 = mtlr_risk(pred_prob3).detach().numpy()  
        ci_event3  = concordance_index(true_time3, -pred_risk3, event_observed=true_event3)
        
        pred_prob4   = torch.cat([x["preds4"] for x in outputs]).cpu() 
        true_time4   = torch.cat([x["labels"]["time4"] for x in outputs]).cpu()
        true_event4  = torch.cat([x["labels"]["event4"] for x in outputs]).cpu()
        pred_risk4 = mtlr_risk(pred_prob4).detach().numpy()  
        ci_event4  = concordance_index(true_time4, -pred_risk4, event_observed=true_event4)

        print('********************************************************')

        print('val loss is ',loss)
        print('val OS ci_event is ',ci_event1)
        print('val LFFS ci_event is ',ci_event2)
        print('val RFFS ci_event is ',ci_event3)
        print('val DFFS ci_event is ',ci_event4)

        log = {"val/loss": loss,
               "val/OS_CI": ci_event1,
               "val/LFFS_CI": ci_event2,
               "val/RFFS_CI": ci_event3,
               "val/DFFS_CI": ci_event4,
               }  
        self.log_dict(log)


        PatientID  = [x['labels']['ID'] for x in outputs] 
        PatientID = sum(PatientID, []) #inefficient way to flatten a list


        results = pd.DataFrame({'ID':PatientID, 'OS_risk':pred_risk1, 'OS':true_time1,'Death':true_event1,\
                                'LFFS_risk':pred_risk2, 'LFFS':true_time2,'LF':true_event2,\
                                'RFFS_risk':pred_risk3, 'RFFS':true_time3,'RF':true_event3,\
                                'DFFS_risk':pred_risk4, 'DFFS':true_time4,'DF':true_event4})
        results.to_csv('Predictions.csv')

        return {"loss": loss, "OS-CI": ci_event1, "LFFS-CI": ci_event2, "RFFS-CI": ci_event3, "DFFS-CI": ci_event4}



    # def test_step(self, batch: Any, batch_idx: int):
        
    #     loss, preds, y, labels, pred_mask, target_mask = self.step(batch)
    #     return {"loss": loss, "preds": preds, "y": y, "labels": labels, "pred_mask": pred_mask, "target_mask": target_mask}



    # def test_epoch_end(self, outputs: List[Any]):
    #     loss        = torch.stack([x["loss"] for x in outputs]).mean()
    #     pred_prob   = torch.cat([x["preds"] for x in outputs]).cpu() 
    #     y           = torch.cat([x["y"] for x in outputs]).cpu()
    #     pred_mask   = torch.cat([x["pred_mask"] for x in outputs])
    #     target_mask   = torch.cat([x["target_mask"] for x in outputs])

    #     true_time   = torch.cat([x["labels"]["time"] for x in outputs]).cpu()
    #     true_event  = torch.cat([x["labels"]["event"] for x in outputs]).cpu()


    #     pred_risk = mtlr_risk(pred_prob).detach().numpy()  

    #     ci_event  = concordance_index(true_time, -pred_risk, event_observed=true_event)

    #     log = {"val/loss": loss,
    #            "val/ci": ci_event,
    #            "val/dice": dice(pred_mask, target_mask),
            
    #            }
        
    #     self.log_dict(log)

        
    #     logits  = torch.cat([x["preds"] for x in outputs]).cpu()
    #     pred_risk = mtlr_risk(logits).detach().numpy()


    #     PatientID  = [x['labels']['PatientID'] for x in outputs] 
    #     PatientID = sum(PatientID, []) #inefficient way to flatten a list


    #     results = pd.DataFrame({'PatientID':PatientID, 'pred_risk':pred_risk})
    #     results.to_csv('Predictions.csv')
        


    #     return {"loss": loss, "CI": ci_event}
   

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = make_optimizer(AdamW, self.model, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # optimizer = make_optimizer(SGD, self.model, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # optimizer = make_optimizer(Adam, self.model, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # optimizer = make_optimizer(Adamax, self.model, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        

        # scheduler = {
        #     "scheduler": MultiStepLR(optimizer, milestones=[self.hparams.step], gamma=0.1),
          
        #     "monitor": "val/loss",
        # }
        
        
        scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, mode='min', patience=5, threshold=0.001, factor=0.1, verbose=True),
          
            "monitor": "val/loss",
        }

        return [optimizer] , [scheduler]
    
    

def make_optimizer(opt_cls, model, **kwargs):
    """Creates a PyTorch optimizer for MTLR training."""
    params_dict = dict(model.named_parameters())
    weights = [v for k, v in params_dict.items() if "mtlr" not in k and "bias" not in k]
    biases = [v for k, v in params_dict.items() if "bias" in k]
    mtlr_weights = [v for k, v in params_dict.items() if "mtlr_weight" in k]
    # Don't use weight decay on the biases and MTLR parameters, which have
    # their own separate L2 regularization
    optimizer = opt_cls([
        {"params": weights},
        {"params": biases, "weight_decay": 0.},
        {"params": mtlr_weights, "weight_decay": 0.},
    ], **kwargs)
    return optimizer




class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1

    def forward(self, input, target):
        axes = tuple(range(1, input.dim()))
        intersect = (input * target).sum(dim=axes)
        union = torch.pow(input, 2).sum(dim=axes) + torch.pow(target, 2).sum(dim=axes)
        loss = 1 - (2 * intersect + self.smooth) / (union + self.smooth)
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = 1e-3

    def forward(self, input, target):
        input = input.clamp(self.eps, 1 - self.eps)
        loss = - (target * torch.pow((1 - input), self.gamma) * torch.log(input) +
                  (1 - target) * torch.pow(input, self.gamma) * torch.log(1 - input))
        return loss.mean()


class Dice_and_FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(Dice_and_FocalLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.dice_loss(input, target) + self.focal_loss(input, target)
        return loss
    

def dice(input, target):
    axes = tuple(range(1, input.dim()))
    bin_input = (input > 0.5).float()

    intersect = (bin_input * target).sum(dim=axes)
    union = bin_input.sum(dim=axes) + target.sum(dim=axes)
    score = 2 * intersect / (union + 1e-3)

    return score.mean()


def hausdorff_distance(image0, image1):
    """Code copied from 
    https://github.com/scikit-image/scikit-image/blob/main/skimage/metrics/set_metrics.py#L7-L54
    for compatibility reason with python 3.6
    """
    a_points = np.transpose(np.nonzero(image0.cpu()))
    b_points = np.transpose(np.nonzero(image1.cpu()))

    # Handle empty sets properly:
    # - if both sets are empty, return zero
    # - if only one set is empty, return infinity
    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf

    return max(max(cKDTree(a_points).query(b_points, k=1)[0]),
               max(cKDTree(b_points).query(a_points, k=1)[0]))
import sys
import numpy as np
import inspect
import importlib
import random
import pandas as pd

#---->
from MyOptimizer import create_optimizer
from MyLoss import create_loss
from utils.utils import cross_entropy_torch

#---->
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

#---->
import pytorch_lightning as pl
from sklearn import metrics
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt




class  ModelInterface(pl.LightningModule):

    #---->init
    def __init__(self, model, loss, optimizer, **kargs):
        super(ModelInterface, self).__init__()
        self.save_hyperparameters()
        self.load_model()
        self.loss = create_loss(loss)
        self.optimizer = optimizer
        self.n_classes = model.n_classes
        self.log_path = kargs['log']
        self.asynchronous = kargs['asynchronous']
        self.save_log = []
        #---->acc
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.epoch = 0
        #---->Metrics
        if self.n_classes > 2: 
            self.AUROC = torchmetrics.AUROC(num_classes = self.n_classes, average = 'macro', task='multiclass')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes = self.n_classes,
                                                                           average='micro', task='multiclass'),
                                                     torchmetrics.CohenKappa(num_classes = self.n_classes, task='multiclass'),
                                                     torchmetrics.F1Score(num_classes = self.n_classes,
                                                                     average = 'macro', task='multiclass'),
                                                     torchmetrics.Recall(average = 'macro',
                                                                         num_classes = self.n_classes, task='multiclass'),
                                                     torchmetrics.Precision(average = 'macro',
                                                                            num_classes = self.n_classes, task='multiclass'),
                                                     torchmetrics.Specificity(average = 'macro',
                                                                            num_classes = self.n_classes, task='multiclass')])
        else : 
            self.AUROC = torchmetrics.AUROC(num_classes=2, task='binary')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes = 2, task='binary'),
                                                     torchmetrics.CohenKappa(num_classes = 2, task='binary'),
                                                     torchmetrics.F1Score(num_classes = 2,
                                                                     task='binary'),
                                                     torchmetrics.Recall(task='binary',
                                                                         num_classes = 2),
                                                     torchmetrics.Precision(task='binary',
                                                                            num_classes = 2)])
        self.valid_metrics = metrics.clone(prefix = 'val_')
        self.test_metrics = metrics.clone(prefix = 'test_')

        #--->random
        self.shuffle = kargs['data'].data_shuffle
        self.count = 0


    #---->remove v_num
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
 
    def training_step(self, batch, batch_idx):
        #---->inference
        #self.epoch = current_epoch
        data, label = batch
        features = data['features']
        reward = data['reward']
        prob, selected_num = self.model.encoder(features.detach().clone()[0], label, reward.detach().clone()[0])
        data_slected = torch.index_select(features, dim = 1, index = selected_num)
        results_dict = self.model(data=data_slected, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        #---->loss
        if self.epoch >= self.asynchronous:
            loss = self.loss(logits, label) + 0.01 * self.loss(prob, label.repeat(prob.size()[0]))
        else:
            loss = self.loss(prob, label.repeat(prob.size()[0]))
        #---->acc log
        Y_hat = int(Y_hat)
        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

        return {'loss': loss} 

    def training_epoch_end(self, training_step_outputs):
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def validation_step(self, batch, batch_idx):
        data, label = batch
        features = data['features']
        reward = data['reward']
        prob, selected_num = self.model.encoder(features.detach().clone()[0], label, reward.detach().clone()[0])
        data_slected = torch.index_select(features, dim = 1, index = selected_num)
        results_dict = self.model(data=data_slected, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']


        #---->acc log
        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}


    def validation_epoch_end(self, val_step_outputs):
        logits = torch.cat([x['logits'] for x in val_step_outputs], dim = 0)
        probs = torch.cat([x['Y_prob'] for x in val_step_outputs], dim = 0)
        max_probs = torch.stack([x['Y_hat'] for x in val_step_outputs])
        target = torch.stack([x['label'][0] for x in val_step_outputs], dim = 0)
        #draw_confusion_matrix(target.cpu().tolist(), max_probs.squeeze(1).cpu().tolist(), 'PEAK_Trans_Dis_L_30_' + str(self.epoch))
        self.save_log.append({'t': target.cpu().tolist(), 'p': max_probs.squeeze(1).cpu().tolist()})
        #---->
        self.log('val_loss', cross_entropy_torch(logits, target), prog_bar=True, on_epoch=True, logger=True)
        self.log('auc', metrics.roc_auc_score(label_binarize(target.unsqueeze(1).cpu().tolist(), classes=[0, 1, 2, 3, 4]), probs.cpu().tolist(), multi_class = 'ovo'), prog_bar=True, on_epoch=True, logger=True)
        self.log_dict(self.valid_metrics(max_probs.squeeze() , target.squeeze()),
                          on_epoch = True, logger = True)
        self.epoch += 1
        #---->acc log
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        
        #---->random, if shuffle data, change seed
        if self.shuffle == True:
            self.count = self.count+1
            random.seed(self.count*50)
    


    def configure_optimizers(self):
        optimizer = create_optimizer(self.optimizer, self.model)
        return [optimizer]

    def test_step(self, batch, batch_idx):
        data, label = batch
        prob, selected_num = self.model.encoder(data.detach().clone()[0])
        data_slected = torch.index_select(data, dim = 1, index = selected_num)
        results_dict = self.model(data=data_slected, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        #---->acc log
        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}

    def test_epoch_end(self, output_results):
        probs = torch.cat([x['Y_prob'] for x in output_results], dim = 0)
        max_probs = torch.stack([x['Y_hat'] for x in output_results])
        target = torch.stack([x['label'] for x in output_results], dim = 0)
        
        #---->
        auc = self.AUROC(probs, target.squeeze())
        metrics = self.test_metrics(max_probs.squeeze() , target.squeeze())
        metrics['auc'] = auc
        for keys, values in metrics.items():
            print(f'{keys} = {values}')
            metrics[keys] = values.cpu().numpy()
        print()
        #---->acc log
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        #---->
        result = pd.DataFrame([metrics])
        result.to_csv(self.log_path / 'result.csv')


    def load_model(self):
        name = self.hparams.model.name
        # Change the `trans_unet.py` file name to `TransUnet` class name.
        # Please always name your model file name as `trans_unet.py` and
        # class name or funciton name corresponding `TransUnet`.
        if '_' in name:
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
        else:
            camel_name = name
        try:
            Model = getattr(importlib.import_module(
                f'models.{name}'), camel_name)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)
        pass

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.model.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.model, arg)
        args1.update(other_args)
        return Model(**args1)
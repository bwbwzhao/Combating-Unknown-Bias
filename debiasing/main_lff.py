import sys
sys.path.append('./')
import os
import pickle
from torch.functional import norm
from tqdm import tqdm
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
from torch.utils.tensorboard import SummaryWriter

from data.util import get_dataset
from utils.loss import GeneralizedCELoss
from module.util import get_model
from utils.util import MultiDimAverageMeter, EMA
import yaml
import argparse
import random


def evaluate(cfg, model, data_loader, attr_dims):
    model.eval()
    attrwise_acc_meter = MultiDimAverageMeter(attr_dims)
    for _, data, attr in data_loader:
        data, attr = data.to(device), attr.to(device)
        label = attr[:, cfg['target_attr_idx']]

        with torch.no_grad():
            logit = model(data)
            pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
            correct = (pred == label).long()

        attr = attr[:, [cfg['target_attr_idx'], cfg['bias_attr_idx']]]
        attrwise_acc_meter.add(correct.cpu(), attr.cpu())

    accs = attrwise_acc_meter.get_mean()

    return accs


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--cfg', default='./configs/mnist.yaml', type=str)
    parser.add_argument('--dataset_tag', default='ColoredMNIST-Skewed0.01-Severity4', type=str)
    parser.add_argument('--main_tag', default='tmp', type=str)
    parser.add_argument('--seed', default=0, type=int)
    args, unparsed = parser.parse_known_args()
    cfg = yaml.safe_load(open(args.cfg))
    cfg['dataset_tag'] = args.dataset_tag
    cfg['main_tag'] = args.main_tag + '_seed' + str(args.seed)
    print(cfg)

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(cfg['device'])
    writer = SummaryWriter(os.path.join(cfg['log_dir'], cfg['dataset_tag'], cfg['main_tag']))
    os.makedirs(os.path.join(cfg['log_dir'], cfg['dataset_tag'], cfg['main_tag']), exist_ok=True)

    # prepare train data
    train_dataset = get_dataset(
        cfg['dataset_tag'],
        data_dir=cfg['data_dir'],
        dataset_split="train",
        transform_split="train",
    )
    train_target_attr = train_dataset.attr[:, cfg['target_attr_idx']]
    train_bias_attr = train_dataset.attr[:, cfg['bias_attr_idx']]
    attr_dims = []
    attr_dims.append(torch.max(train_target_attr).item() + 1)
    attr_dims.append(torch.max(train_bias_attr).item() + 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['main_batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # prepare valid data
    valid_dataset = get_dataset(
        cfg['dataset_tag'],
        data_dir=cfg['data_dir'],
        dataset_split="eval",
        transform_split="eval",
    )
    valid_target_attr = valid_dataset.attr[:, cfg['target_attr_idx']]
    valid_bias_attr = valid_dataset.attr[:, cfg['bias_attr_idx']]
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg['main_batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # prepare model and optimizer
    model_b = get_model(cfg['model_tag'], attr_dims[0]).to(device)
    model = get_model(cfg['model_tag'], attr_dims[0]).to(device)
    optimizer_b = torch.optim.Adam(
        model_b.parameters(),
        lr=cfg['main_learning_rate'],
        weight_decay=cfg['main_weight_decay'],
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg['main_learning_rate'],
        weight_decay=cfg['main_weight_decay'],
    )
    
    # define loss
    criterion = nn.CrossEntropyLoss(reduction='none')
    bias_criterion = GeneralizedCELoss(q=0.7)
    sample_loss_ema_b = EMA(torch.LongTensor(train_target_attr), alpha=0.7)
    sample_loss_ema_d = EMA(torch.LongTensor(train_target_attr), alpha=0.7)

    count_iter = 0
    best_valid = 0.
    list_valid, list_valid_aligned, list_valid_skewed = [], [], []
    for epoch in range(cfg['main_num_epoch']):
        print('epoch', epoch)
        # train
        model_b.train()
        model.train()
        for _ in range(len(train_dataset)//cfg['main_batch_size']):
            try:
                index, data, attr = next(train_iter)
            except:
                train_iter = iter(train_loader)
                index, data, attr = next(train_iter)
            data, attr = data.to(device), attr.to(device)
            label = attr[:, cfg['target_attr_idx']]
            bias_attr = attr[:, cfg['bias_attr_idx']]
            aligned_mask = (label == bias_attr)
            skewed_mask = (label != bias_attr)

            logit_b = model_b(data)
            logit_d = model(data)
            
            loss_b = criterion(logit_b, label).cpu().detach()
            loss_d = criterion(logit_d, label).cpu().detach()
            # EMA sample loss
            sample_loss_ema_b.update(loss_b, index)
            sample_loss_ema_d.update(loss_d, index)
            # class-wise normalize
            loss_b = sample_loss_ema_b.parameter[index].clone().detach()
            loss_d = sample_loss_ema_d.parameter[index].clone().detach()
            label_cpu = label.cpu()
            for c in range(attr_dims[0]):
                class_index = np.where(label_cpu == c)[0]
                max_loss_b = sample_loss_ema_b.max_loss(c)
                max_loss_d = sample_loss_ema_d.max_loss(c)
                loss_b[class_index] /= max_loss_b
                loss_d[class_index] /= max_loss_d
                
            # re-weighting based on loss value / generalized CE for biased model
            loss_weight = loss_b / (loss_b + loss_d + 1e-8)
            loss_b_update = bias_criterion(logit_b, label)
            loss_d_update = criterion(logit_d, label) * loss_weight.to(device)
            loss = loss_b_update.mean() + loss_d_update.mean()
            
            optimizer_b.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer_b.step()
            optimizer.step()

            count_iter += 1

        # test
        eye_tsr = torch.eye(attr_dims[0])
        valid_attrwise_accs = evaluate(cfg, model, valid_loader, attr_dims)
        valid_accs = torch.mean(valid_attrwise_accs)
        writer.add_scalar("acc/valid", valid_accs, epoch)
        writer.add_scalar("acc/valid_aligned", valid_attrwise_accs[eye_tsr == 1.0].mean(), epoch)
        writer.add_scalar("acc/valid_skewed", valid_attrwise_accs[eye_tsr == 0.0].mean(), epoch)
        list_valid.append(valid_accs.item())
        list_valid_aligned.append(valid_attrwise_accs[eye_tsr == 1.0].mean().item())
        list_valid_skewed.append(valid_attrwise_accs[eye_tsr == 0.0].mean().item())
        if valid_accs > best_valid:
            # save best
            model_path = os.path.join(cfg['log_dir'], cfg['dataset_tag'], cfg['main_tag'], "model_best.th")
            state_dict = {
                'steps': epoch, 
                'state_dict': model.state_dict(), 
                'optimizer': optimizer.state_dict(), 
                'accs': valid_attrwise_accs
            }
            torch.save(state_dict, model_path)
            
            best_valid = valid_accs.item()
            
    # record best
    best_e = np.argmax(list_valid)
    writer.add_scalar("best_acc/valid", list_valid[best_e], best_e)
    writer.add_scalar("best_acc/valid_aligned", list_valid_aligned[best_e], best_e)
    writer.add_scalar("best_acc/valid_skewed", list_valid_skewed[best_e], best_e)
    # save last
    model_path = os.path.join(cfg['log_dir'], cfg['dataset_tag'], cfg['main_tag'], "model_%d.th"%(epoch))
    state_dict = {
        'steps': epoch, 
        'state_dict': model.state_dict(), 
        'optimizer': optimizer.state_dict(), 
        'accs': valid_attrwise_accs
    }
    torch.save(state_dict, model_path)

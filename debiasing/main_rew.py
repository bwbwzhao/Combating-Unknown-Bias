import sys
sys.path.append('./')
import os
import pickle
from tqdm import tqdm
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from torch.utils.tensorboard import SummaryWriter

from data.util import get_dataset
from module.util import get_model
from utils.util import MultiDimAverageMeter, EMA
import yaml
import argparse
import matplotlib.pyplot as plt
from data.attr_dataset import AttributeDataset
import copy
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
    parser.add_argument('--gamma', default=1.0, type=float)
    args, unparsed = parser.parse_known_args()
    cfg = yaml.safe_load(open(args.cfg))
    cfg['dataset_tag'] = args.dataset_tag
    cfg['gamma'] = args.gamma
    cfg['main_tag'] = args.main_tag + '_gamma' + str(cfg['gamma']) + '_seed' + str(args.seed)
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
    
    hard_set = copy.deepcopy(train_dataset)
    hard_set.attr = hard_set.attr[train_target_attr!=train_bias_attr]
    hard_set.data = hard_set.data[train_target_attr!=train_bias_attr]
    easy_set = copy.deepcopy(train_dataset)
    easy_set.attr = easy_set.attr[train_target_attr==train_bias_attr]
    easy_set.data = easy_set.data[train_target_attr==train_bias_attr]
    hard_bs = int(len(hard_set) / len(train_dataset) * cfg['main_batch_size']) + 1
    easy_bs = cfg['main_batch_size'] - hard_bs
    hard_train_loader = DataLoader(
        hard_set,
        batch_size=hard_bs,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    easy_train_loader = DataLoader(
        easy_set,
        batch_size=easy_bs,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
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
    model = get_model(cfg['model_tag'], attr_dims[0]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg['main_learning_rate'],
        weight_decay=cfg['main_weight_decay'],
    )

    count_iter = 0.
    best_valid = 0.
    list_valid, list_valid_aligned, list_valid_skewed = [], [], []
    for epoch in range(cfg['main_num_epoch']):
        print('epoch', epoch)
        # train
        model.train()
        for _ in range(len(train_dataset)//cfg['main_batch_size']):
            try:
                hard_index, hard_data, hard_attr = next(hard_train_iter)
            except:
                hard_train_iter = iter(hard_train_loader)
                hard_index, hard_data, hard_attr = next(hard_train_iter)

            try:
                easy_index, easy_data, easy_attr = next(easy_train_iter)
            except:
                easy_train_iter = iter(easy_train_loader)
                easy_index, easy_data, easy_attr = next(easy_train_iter)

            data = torch.cat((hard_data, easy_data))
            attr = torch.cat((hard_attr, easy_attr))

            data, attr = data.to(device), attr.to(device)
            label = attr[:, cfg['target_attr_idx']]
            bias_attr = attr[:, cfg['bias_attr_idx']]
            aligned_mask = (label == bias_attr)
            skewed_mask = (label != bias_attr)

            logit = model(data)
            loss_per_sample = F.cross_entropy(logit, label, reduction='none')

            loss_weight = torch.ones_like(loss_per_sample)
            loss_weight[aligned_mask] = skewed_mask.sum() / (aligned_mask.sum() * cfg['gamma'])
            loss_weight = loss_weight / (loss_weight.sum() + 1e-10) * len(loss_weight)

            loss = (loss_per_sample * loss_weight).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count_iter+=1

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

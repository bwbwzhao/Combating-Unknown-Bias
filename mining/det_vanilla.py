import sys
sys.path.append('./')
import os
import numpy as np
from numpy.core.records import record
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.util import get_dataset
from module.util import get_model
from utils.util import MultiDimAverageMeter
import yaml
import argparse
import random
import copy


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--cfg', default='./configs/mnist.yaml', type=str)
    parser.add_argument('--dataset_tag', default='ColoredMNIST-Skewed0.01-Severity4', type=str)
    parser.add_argument('--main_tag', default='tmp', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--eta', default=0.0, type=float)
    args, unparsed = parser.parse_known_args()
    cfg = yaml.safe_load(open(args.cfg))
    cfg['dataset_tag'] = args.dataset_tag
    cfg['eta'] = args.eta
    cfg['main_tag'] = args.main_tag + '_eta' + str(args.eta)
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

    # prepare train_test data
    train_test_dataset = copy.deepcopy(train_dataset)
    train_test_dataset.transform = valid_dataset.transform
    train_test_loader = DataLoader(
        train_test_dataset,
        batch_size=cfg['main_batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # prepare model and optimizer
    model_b = get_model(cfg['model_tag'], attr_dims[0]).to(device)
    optimizer_b = torch.optim.Adam(
        model_b.parameters(),
        lr=cfg['main_learning_rate'],
        weight_decay=cfg['main_weight_decay'],
    )

    count_iter = 0
    scores = []
    for epoch in range(50):
        print('epoch', epoch)
        # train
        model_b.train()
        for _ in range(len(train_dataset)//cfg['main_batch_size']):
            try:
                _, data, attr = next(train_iter)
            except:
                train_iter = iter(train_loader)
                _, data, attr = next(train_iter)
            data, attr = data.to(device), attr.to(device)
            label = attr[:, cfg['target_attr_idx']]
            bias_attr = attr[:, cfg['bias_attr_idx']]
            aligned_mask = (label == bias_attr)
            skewed_mask = (label != bias_attr)

            logit = model_b(data)
            loss_per_sample = F.cross_entropy(logit, label, reduction='none')
            loss = loss_per_sample.mean()
            optimizer_b.zero_grad()
            loss.backward()
            optimizer_b.step()
            count_iter += 1

        # test
        model_b.eval()
        attrwise_acc_meter = MultiDimAverageMeter(attr_dims)
        score = []
        for _, data, attr in train_test_loader:
            data, attr = data.to(device), attr.to(device)
            label = attr[:, cfg['target_attr_idx']]
            with torch.no_grad():
                logit = model_b(data)
                loss = F.cross_entropy(logit, label, reduction='none')
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()
                score.append(torch.exp(-loss))
            attr = attr[:, [cfg['target_attr_idx'], cfg['bias_attr_idx']]]
            attrwise_acc_meter.add(correct.long().cpu(), attr.cpu())
        attrwise_accs = attrwise_acc_meter.get_mean()
        eye_tsr = torch.eye(attr_dims[0])
        writer.add_scalar("b_acc/train_aligned", attrwise_accs[eye_tsr == 1.0].mean(), epoch)
        writer.add_scalar("b_acc/train_skewed", attrwise_accs[eye_tsr == 0.0].mean(), epoch)
        score = torch.cat(score)
        scores.append(score)
        avg_scores = torch.stack(scores).sum(dim=0) / (epoch+1)
        
        writer.add_pr_curve(
            'ensemble', 
            train_target_attr!=train_bias_attr, 
            1. - avg_scores, 
            epoch,
            len(avg_scores)
        )
        writer.add_pr_curve(
            'single', 
            train_target_attr!=train_bias_attr, 
            1. - score, 
            epoch,
            len(score)
        )

    # save
    model_path = os.path.join(cfg['log_dir'], cfg['dataset_tag'], cfg['main_tag'], "model_b.th")
    state_dict = {
        'scores': torch.stack(scores).t().cpu(),
        'train_target_attr': train_target_attr.cpu(),
        'train_bias_attr': train_bias_attr.cpu(),
    }
    torch.save(state_dict, model_path)
import torch.optim as optim
from torch import nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, confusion_matrix, fbeta_score, \
    precision_score, recall_score
import torch
import numpy as np
import math
import copy
import torch.nn.functional as F
import sys


def parse_optimizer(parser):
    opt_parser = parser.add_argument_group()
    opt_parser.add_argument('--opt', dest='opt', type=str,
                            help='Type of optimizer')
    opt_parser.add_argument('--opt-scheduler', dest='opt_scheduler', type=str,
                            help='Type of optimizer scheduler. By default none')
    opt_parser.add_argument('--opt-restart', dest='opt_restart', type=int,
                            help='Number of epochs before restart (by default set to 0 which means no restart)')
    opt_parser.add_argument('--opt-decay-step', dest='opt_decay_step', type=int,
                            help='Number of epochs before decay', default=50)
    opt_parser.add_argument('--opt-decay-rate', dest='opt_decay_rate', type=float,
                            help='Learning rate decay ratio', default=0.8)
    opt_parser.add_argument('--lr', dest='lr', type=float,
                            help='Learning rate.')
    opt_parser.add_argument('--clip', dest='clip', type=float,
                            help='Gradient clipping.')
    opt_parser.add_argument('--weight_decay', type=float,
                            help='Optimizer weight decay.', default=0)


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 300, 500, 700, 900],
                                                   gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer



class Logger(object):
    def __init__(self, dir):
        self.terminal = sys.stdout
        self.log = open(f"{dir}/log.dat", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def CE_loss(pred, label):
    label = label.type(torch.int64)
    loss = nn.CrossEntropyLoss()
    return loss(pred, label)

def BCE_loss(pred, label):
    pred = pred.view(-1)
    label = label.view(-1).type(torch.float32)
    loss = nn.BCELoss()
    return loss(pred, label)

def get_acc_score(pred, label):
    pred_label = pred.detach().clone()
    pred_label = pred_label.argmax(dim=1)
    acc_score = accuracy_score(label.cpu().detach().numpy(), pred_label.cpu().detach().numpy())
    return acc_score


def get_auc_score(pred, label):
    try:
        pred = F.softmax(pred, dim=1)
        if torch.unique(label).size(0) > 2:
            auc_score = roc_auc_score(label.cpu().detach().numpy(), pred.cpu().detach().numpy(), multi_class='ovr')
        else:
            auc_score = roc_auc_score(label.cpu().detach().numpy(), pred[:, 1].cpu().detach().numpy())
        return auc_score
    except ValueError:
        print("auc error")
        return None
        pass


def get_scores(pred, label, name=''):
    auc_score = 1
    acc_score = get_acc_score(pred, label)
    
    if isinstance(name, (list, tuple)):
        name_str = ','.join(map(str, name))
    else:
        name_str = str(name)
    print(name_str + ' result:')
    print("acc score: " + str(round(acc_score,4)) )
    print()


    return acc_score, auc_score

def cal_str_dif(pred_mtx, true_mtx):
    cls1_diff = torch.abs(pred_mtx - true_mtx)
    cls0_diff = torch.abs((1 - pred_mtx) - (1 - true_mtx))
    return 0.5 * torch.sum(cls0_diff) + 0.5 * torch.sum(cls1_diff)

def cal_str_dif_rel(pred_mtx, true_mtx):
    cls1_diff = torch.abs(pred_mtx - true_mtx)
    cls0_diff = torch.abs((1 - pred_mtx) - (1 - true_mtx))
    abs_diff = 0.5 * cls0_diff + 0.5 * cls1_diff
    rel_diff_1 = abs_diff / true_mtx
    rel_diff_2 = abs_diff / pred_mtx
    rel_diff = 0.5 * rel_diff_1 + 0.5 * rel_diff_2
    rel_diff[torch.isinf(rel_diff_1)] = rel_diff_2[torch.isinf(rel_diff_1)]
    rel_diff[torch.isinf(rel_diff_2)] = rel_diff_1[torch.isinf(rel_diff_2)]
    rel_diff[torch.isnan(rel_diff)] = 0

    num = true_mtx.size(0) * true_mtx.size(1)
    return torch.sum(abs_diff) / num, torch.sum(rel_diff) / num

def cal_str_dif_log(pred_mtx, true_mtx):
    ratio = true_mtx / pred_mtx
    ratio[torch.nonzero(ratio == 0)] = 1
    ratio[torch.isinf(ratio)] = 1
    log_matrix = torch.abs(torch.log(ratio))
    num = true_mtx.size(0) * true_mtx.size(1)
    return torch.sum(log_matrix) / num

def cal_str_diff_ratio(pred_mtx, true_mtx):
    intra_prob_pred = torch.diagonal(pred_mtx, 0).repeat_interleave(pred_mtx.size(1)).view(-1, pred_mtx.size(1))
    intra_prob_true = torch.diagonal(true_mtx, 0).repeat_interleave(true_mtx.size(1)).view(-1, true_mtx.size(1))
    pred_ratio = torch.div(pred_mtx, intra_prob_pred)
    true_ratio = torch.div(true_mtx, intra_prob_true)
    pred_ratio[torch.isnan(pred_ratio)] = 1
    true_ratio[torch.isnan(true_ratio)] = 1
    pred_ratio[torch.isinf(pred_ratio)] = pred_mtx[torch.isinf(pred_ratio)]
    true_ratio[torch.isinf(true_ratio)] = true_mtx[torch.isinf(true_ratio)]

    ratio_diff = torch.div(pred_ratio, true_ratio)
    ratio_diff[torch.isnan(ratio_diff)] = 1
    ratio_diff[torch.isinf(ratio_diff)] = 1

    num = true_mtx.size(0) * true_mtx.size(1) - pred_mtx.size(0)
    return (torch.sum(ratio_diff) - torch.sum(torch.diagonal(ratio_diff))) / num


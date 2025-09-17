import os
import sys
import socket
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import warnings

warnings.filterwarnings("ignore")

conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/datasets')
sys.path.append(conf_path + '/backbone')
sys.path.append(conf_path + '/models')
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from datasets import Priv_NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import get_prive_dataset
from models import get_model
from utils.training_graph import train
from utils.best_args import best_args
from utils.conf import set_random_seed
import setproctitle
import torch
import uuid
import datetime
from datasets.utils.federated_dataset import FederatedDataset
import torch, gc
from itertools import product

def parse_args():
    parser = ArgumentParser(description='You Only Need Me', allow_abbrev=False)
    parser.add_argument('--device_id', type=int, default=0, help='The Device Id for Experiment')
    parser.add_argument('--communication_epoch', type=int, default=200, help='The Communication Epoch in Federated Learning')

    parser.add_argument('--dataset', type=str, default='fl_twitch', 
                        choices=DATASET_NAMES, help='Which scenario to perform experiments on.')

    parser.add_argument('--data_root', type=str, default='./datasets/data', help='data root')
    parser.add_argument('--backbone', default='pmlp_gcn', type=str, help='Backbone')

    parser.add_argument('--model', type=str, default='fedavgg',  
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--proposed', type=bool, default=False)
    parser.add_argument('--project', type=str, default='spec', choices=['spec', 'svd'])
    parser.add_argument('--similarity', type=str, default='cos', choices=['cos', 'cka'])
    parser.add_argument('--mask_ratio', type=float, default=0.1)
    parser.add_argument('--delta_beta', type=float, default=0.1,
                        help='Momentum weight update coefficient')
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--omega', type=float, default=5)
    parser.add_argument('--lambda_c', type=float, default=1)
    parser.add_argument('--lambda_d', type=float, default=1)
    parser.add_argument('--proto_momentum', type=float, default=0.9)
    parser.add_argument('--proto_temp', type=float, default=0.07)
    parser.add_argument('--proto_loss_weight', type=float, default=0.5)

    parser.add_argument('--use_knn', action='store_true', default=True, help='knn for evaluating')

    parser.add_argument('--hidden', type=int, default=128, help='The backbone hidden layer')

    parser.add_argument('--knn', type=int, default=30, help='Nearest neighbors for evaluate')

    parser.add_argument('--size', type=int, default=10, help='queue_size')
    parser.add_argument('--neibor', type=int, default=3, help='neibor')

    parser.add_argument('--personal', type=str, default='lklk', help='personal layer')

    parser.add_argument('--knn_frequence', type=int, default=20, help='Compute knn_frequence')
    parser.add_argument('--ema', type=float, default=0.9, help='ema for momentum update')


    parser.add_argument('--use_att', type=bool, default=False, help='use conv for training')
    parser.add_argument('--train_conv', type=bool, default=True, help='use conv for training')
    parser.add_argument('--local_epoch', type=int, default=5, help='The Local Epoch for each Participant')
    parser.add_argument('--local_lr', type=float, default=0.005)


    parser.add_argument('--parti_num', type=int, default=10, help='The Number for Participants') 
    parser.add_argument('--depth', type=int, default=2, help='The backbone depth')
    parser.add_argument('--seed', type=int, default=0, help='The random seed.')
    parser.add_argument('--rand_dataset', type=bool, default=False, help='The random seed.')


    parser.add_argument('--structure', type=str, default='homogeneity')

    parser.add_argument('--pri_aug', type=str, default='weak',  
                        help='Augmentation for Private Data')
    parser.add_argument('--online_ratio', type=float, default=1, help='The Ratio for Online Clients')
    parser.add_argument('--learning_decay', type=bool, default=False, help='The Option for Learning Rate Decay')
    parser.add_argument('--averaging', type=str, default='weight', help='The Option for averaging strategy')
    parser.add_argument('--mu', type=float, default=0.01, help='')

    parser.add_argument('--infoNCET', type=float, default=0.02, help='The InfoNCE temperature')
    parser.add_argument('--T', type=float, default=0.05, help='The Knowledge distillation temperature')
    parser.add_argument('--weight', type=int, default=1, help='The Weight for the distillation loss')
    parser.add_argument('--mu_moon', type=float, default=5, help='The Weight for the distillation loss')
    parser.add_argument('--temperature_moon', type=float, default=0.5, help='The Weight for moon')
    parser.add_argument('--reserv_ratio', type=float, default=0.1, help='Reserve ratio for prototypes')


    torch.set_num_threads(4)
    add_management_args(parser)
    args = parser.parse_args()


    if args.seed is not None:
        set_random_seed(args.seed)
    return args


def main(args=None):
    if args is None:
        args = parse_args()
    if args.dataset == 'fl_arxiv':
            args.knn_frequence = 50
            args.neibor = 1
            args.knn = 100
    if args.dataset == 'fl_citation':
            args.knn_frequence = 10
            args.neibor = 3
            args.knn = 20
    if args.dataset in ['fl_twitch', 'fl_wikinet']:
            args.knn_frequence = 10
            args.neibor = 3
            args.knn = 40
            args.size = 20
    if args.dataset == 'fl_jodie':
            args.knn_frequence = 10
            args.neibor = 3
            args.knn = 20

    if args.use_knn:
        args.personal = 'fcs.2'

    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    print("Initializing federated dataset...")
    priv_dataset = get_prive_dataset(args)
    if not isinstance(priv_dataset, FederatedDataset):
        raise TypeError("Dataset must inherit from FederatedDataset")

    print("Loading data...")
    train_loaders, test_loaders = priv_dataset.get_data_loaders()
    args.parti_num = priv_dataset.get_parti()
    model_list = [args.backbone for _ in range(args.parti_num)]
    print('load backbone:' + str(args.backbone))
    print('use_knn:' + str(args.use_knn))
    print('personal_layer: ' + str(args.personal))
    backbones_list = priv_dataset.get_backbone(args.parti_num, model_list)
    model = get_model(backbones_list, args, priv_dataset.get_transform())

    print('{}_{}_{}_{}_{}'.format(args.model, args.parti_num, args.dataset, args.communication_epoch, args.local_epoch))
    setproctitle.setproctitle('{}_{}_{}_{}_{}'.format(args.model, args.parti_num, args.dataset, args.communication_epoch, args.local_epoch))

    train(model, priv_dataset, args)

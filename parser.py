import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=2e-3, type=float, help='learning rate')
    parser.add_argument('--decay', default=0.99, type=float, help='learning rate')
    parser.add_argument('--batch', default=128, type=int, help='batch size')
    parser.add_argument('--inter_batch', default=4096, type=int, help='batch size')
    parser.add_argument('--note', default=None, type=str, help='note')
    parser.add_argument('--lambda1', default=0.2, type=float, help='weight of cl loss')
    parser.add_argument('--lambda3', default=1.5, type=float, help='weight of recommendation loss')
    parser.add_argument('--epoch', default=150, type=int, help='number of epochs')
    parser.add_argument('--d', default=256, type=int, help='embedding size')
    parser.add_argument('--q', default=5, type=int, help='rank')
    parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
    parser.add_argument('--data', default='yelp', type=str, help='name of dataset')
    parser.add_argument('--dropout', default=0.0, type=float, help='rate for edge dropout')
    parser.add_argument('--temp', default=0.2, type=float, help='temperature in cl loss')
    parser.add_argument('--lambda2', default=1e-7, type=float, help='l2 reg weight')
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), type=str, help='the gpu to use')
    parser.add_argument('--path',default='./KDHR/',help='path of data,choose in  Set2Set and KDHR')
    # parser.add_argument('--batch_size',default=512,type=int,help='path of data')
    return parser.parse_args()
args = parse_args()
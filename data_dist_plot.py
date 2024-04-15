import numpy as np
import matplotlib
import torch
from sklearn.model_selection import train_test_split
import copy

from utils import load_data
from Dataset import dataset2d
from ModelClass import ModelClass
matplotlib.rcParams['figure.autolayout'] = True
import matplotlib.pyplot as plt

import argparse

import os
if not os.path.exists('Results'):
    os.mkdir('Results')

def int_list(string):
    return sorted([int(s) for s in string.split(',')])


parser = argparse.ArgumentParser()
# Choosing the benchmark:
parser.add_argument('-bench', '--benchmark', type=int, default=0,
                    choices = [0,1,2,3,4,5,6,7,8], help='Benchmark Index')
parser.add_argument('-rand_split', '--rand_split', action='store_true')

# Specifications about the convolutional autoencoder
parser.add_argument('-down', '--down', type=int, default=4,
                    choices=[2,3,4,5], help='Number of Encoder/Decoder stages')
parser.add_argument('-embed_dim', '--embed_dim', type=int, default=128,
                    choices=[32,64,128,256,512,1024],
                    help='feature dimension og the encodings')


parser.add_argument('-dts', '--dts', default='1,2,4')
parser.add_argument('-rollout', '--rollout', type=int, default=1)

# Specifications about the transformer
parser.add_argument('-heads', '--heads', type=int_list, default='8,8,8,8',
                    help='Number of heads through the transformer layers in the form 4,8,8,4')
parser.add_argument('-split', '--split_heads', action='store_true')

parser.add_argument('-attn', '--attn_method', type=int, default=1,
                    choices=[0,1,2,3,4,5,6],
                    help='The attention method')

parser.add_argument('-no_softmax', '--no_softmax', action='store_true')

parser.add_argument('-pos', '--pos_type', type=int, default=1,
                    choices=[0,1,2,3,4,5,6], help='position embedding type')

parser.add_argument('-rng', '--attn_rng', default=10, type=int,
                    help='attention range for custom learnable pos embeddings')

parser.add_argument('-norm', '--norm', action='store_true')
parser.add_argument('-no_mlp', '--no_mlp', action='store_true')

# Specifications about training
parser.add_argument('-batch', '--batch_size', type=int, default=64)
parser.add_argument('-ae_epochs', '--ae_epochs', type=int, default=0)
parser.add_argument('-epochs', '--epochs', type=int, default=1)
parser.add_argument('-with_ae', '--with_ae', action='store_true')

args = parser.parse_args()

Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------------------------------------------------

benchmark = args.benchmark
if benchmark == 1:
    data_name = 'NS_V1e-3_N1200_T50'
elif benchmark == 2:
    data_name = 'NS_V1e-4_N1200_T30'
elif benchmark == 3:
    data_name = 'NS_V1e-5_N1200_T20'

elif benchmark == 7:
    data_name = 'KF_Re40_N200_T500'

if benchmark>0:
    data_array = load_data(data_name)
else:
    data_array = np.random.randn(192,11,64,64).astype(np.float32)


N, T, H, W = data_array.shape

data_array /= max(data_array.max(), -data_array.min())

if args.rand_split:
    train_idx, val_idx = train_test_split(
        np.arange(N),
        test_size = 1/6 if benchmark<7 else 1/5,
        random_state = 0)
else:
    test_size = N//6 if benchmark<7 else N//5
    train_idx, val_idx = np.arange(N)[:-test_size], np.arange(N)[-test_size:]

name = f'bench{benchmark}_dt{args.dts}_r{args.rollout}'
model_idx = len([m for m in os.listdir('Results') if  m.startswith(name)]) + 1
name += f'_model{model_idx}'
model = ModelClass(name, first=True)
model.set_datasets(dataset2d(data_array, train_idx),
                   dataset2d(data_array, val_idx))

# LOADING OR TRAINING THE AUTOENCODER ========================================

u = data_array
du = u[:, 1:, :, :] - u[:, :-1, :, :]
du_norm = torch.norm(du, dim=(-1, -2))
u_norm = torch.norm(u[:, :-1, :, :], dim=(-1, -2))
du_normalized = du_norm / u_norm
du_normalized = du_normalized
plt.figure()
plt.hist(du_normalized[train_idx, ...].flatten().numpy(), bins=100, color='blue', label='train')
plt.hist(du_normalized[val_idx, ...].flatten().numpy(), bins=100, color='red', label='test')
plt.xlabel(r'$\dfrac{|u_{t+1}-u_t|}{|u_t|}$', fontsize=16)
plt.ylabel('Number of data points', fontsize=16)
dataset_name = f'NS{args.benchmark}' if args.benchmark < 7 else 'KF'
plt.title(f'Normalized rate of change per time-step for {dataset_name} dataset', fontsize=16)
plt.legend(fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig(f'data{args.benchmark}_hist.png')
plt.close()

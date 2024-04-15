import numpy as np
import matplotlib
import torch
from sklearn.model_selection import train_test_split
import copy

from utils import load_data
from Dataset import dataset2d
from Model import CNN_Encoder, CNN_Decoder, Transformer2D, L2normLoss
from ModelClass import ModelClass
matplotlib.rcParams['figure.autolayout'] = True

import argparse

import os
if not os.path.exists('Results'):
    os.mkdir('Results')

def int_list(string):
    return sorted([int(s) for s in string.split(',')])


parser = argparse.ArgumentParser()
# Choosing the benchmark:
parser.add_argument('-bench', '--benchmark', type=int, default=0,
                    choices = [0,1,2,3,7], help='Benchmark Index')
parser.add_argument('-rand_split', '--rand_split', action='store_true')

# Specifications about the convolutional autoencoder
parser.add_argument('-down', '--down', type=int, default=4,
                    choices=[2,3,4,5], help='Number of Encoder/Decoder stages')
parser.add_argument('-embed_dim', '--embed_dim', type=int, default=128,
                    choices=[32,64,128,256,512,1024],
                    help='feature dimension of the encodings')


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
if args.dummy:
    model_idx='_dummy'
name += f'_model{model_idx}'
model = ModelClass(name, first=True)
model.set_datasets(dataset2d(data_array, train_idx),
                   dataset2d(data_array, val_idx))



# LOADING OR TRAINING THE AUTOENCODER ========================================

if args.ae_epochs==0 and not args.with_ae:
    AE_name = f'AE_bench{args.benchmark}_down{args.down}_dim{args.embed_dim}'
    AE_name += '_rnd' if args.rand_split else ''
    model.load_state_dict(model_name = AE_name)

else:
    channels = [args.embed_dim//2**i for i in range(args.down)]
    channels = [1] + channels[::-1]
    encoder = CNN_Encoder(channels=channels, padding_mode='circular')
    decoder = CNN_Decoder(channels=channels[::-1], padding_mode='circular')
    model.set_AE(encoder, decoder)

if args.ae_epochs>0:
    model.train_AE(criterion = L2normLoss,
                   epochs = args.ae_epochs,
                   batch_size = args.batch_size)

# TRAINING THE DYNAMICS MODEL ================================================

transformer_kwargs = dict(

    dim = args.embed_dim,
    num_heads = args.heads,
    split = args.split_heads,

    attn_method = args.attn_method,
    softmax = not args.no_softmax,
    pos_type = args.pos_type,
    attn_rng = args.attn_rng,

    qkv_bias = True,
    qk_scale = None,
    attn_drop = 0,
    proj_drop = 0,

    use_norm = args.norm,
    use_fc = not args.no_mlp
    )

# set and train dynamic model
model.model_config = transformer_kwargs

dts = int_list(args.dts)

# to initialize lazy modules:
for i, dt in enumerate(dts):

    if i==0:
        transformer = Transformer2D(**transformer_kwargs)
        a = transformer(model.encoder(torch.randn(1,1,64,64).to(Device)).cpu())
        model.set_model(transformer, dt=dt)
    else:
        # transfer learning
        model.set_model(copy.deepcopy(model.models[str(dts[i-1])]), dt=dt)

    if args.with_ae:
        model.train_all(dt = dt,
                        rollout = args.rollout,
                        criterion = L2normLoss,
                        epochs = args.epochs,
                        batch_size = args.batch_size)
    else:
        model.train(dt = dt,
                    rollout = args.rollout,
                    criterion = L2normLoss,
                    epochs = args.epochs,
                    batch_size = args.batch_size)

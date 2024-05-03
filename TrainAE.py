import numpy as np
import matplotlib
from sklearn.model_selection import train_test_split

from utils import load_data
from Dataset import dataset2d
from Model import CNN_Encoder, CNN_Decoder, L2normLoss
from ModelClass import ModelClass
matplotlib.rcParams['figure.autolayout'] = True

import argparse

import os
if not os.path.exists('Results'):
    os.mkdir('Results')

parser = argparse.ArgumentParser()
# Choosing the benchmark:
parser.add_argument('-bench', '--benchmark', type=int, default=0,
                    choices = [0,1,2,3,7])
parser.add_argument('-rand_split', '--rand_split', action='store_true')
# Specifications about the convolutional autoencoder
parser.add_argument('-down', '--down', type=int, default=4)
parser.add_argument('-embed_dim', '--embed_dim', type=int, default=128,
                    choices=[16,32,64,128,256,512])

parser.add_argument('-batch', '--batch_size', type=int, default=64)
parser.add_argument('-epochs', '--epochs', type=int, default=100)

args = parser.parse_args()

benchmark = args.benchmark
if benchmark == 1:
    file_name = 'MST_NS_v1e-3_N1200_T50.npy'
elif benchmark == 2:
    file_name = 'MST_NS_v1e-4_N1200_T30.npy'
elif benchmark == 3:
    file_name = 'MST_NS_v1e-5_N1200_T20.npy'

elif benchmark == 7:
    file_name = 'KFvorticity_Re40_N200_T500.npy'

if benchmark>0:
    data_array = load_data(file_name)
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


channels = [args.embed_dim//2**i for i in range(args.down)]
channels = [1] + channels[::-1]

name = f'AE_bench{benchmark}_down{args.down}_dim{args.embed_dim}'
name += '_rnd' if args.rand_split else ''

if args.dummy:
    name += '_dummy'

model = ModelClass(name, first=True)
model.set_datasets(dataset2d(data_array, train_idx),
                   dataset2d(data_array, val_idx))

encoder = CNN_Encoder(channels=channels, padding_mode='circular')
decoder = CNN_Decoder(channels=channels[::-1], padding_mode='circular')
model.set_AE(encoder, decoder)
model.train_AE(criterion = L2normLoss,
               epochs = args.epochs,
               batch_size = args.batch_size)
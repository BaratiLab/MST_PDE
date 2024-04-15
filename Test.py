import numpy as np
import matplotlib
from sklearn.model_selection import train_test_split

from utils import load_data, plot_result_samples, plot_learning_curves, viz_compare, plot_result_samples_point
from Dataset import dataset2d
from ModelClass import ModelClass
matplotlib.rcParams['figure.autolayout'] = True
from utils import compare_ES


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
                    choices=[16,32,64,128,256,512,1024],
                    help='feature dimension og the encodings')

# Specifications about the transformer

parser.add_argument('-model', '--model', type=int, default=1)
parser.add_argument('-dts', '--dts', default='1,2,4,8')
parser.add_argument('-rollout', '--rollout', type=int, default=1)

parser.add_argument('-use_dts', '--use_dts', default='1,2,4,8', type=int_list,
                    help='which dts to use')
parser.add_argument('-start_end', '--start_end', default='9,100', type=int_list,
                    help='the time period to be predicted by the model')


parser.add_argument('-curves', '--curves', action='store_true')
parser.add_argument('-point', '--point', action='store_true')
parser.add_argument('-final', '--final', action='store_true')
parser.add_argument('-overtime', '--overtime', action='store_true')
parser.add_argument('-ylim', '--ylim', type=int_list, default='0,30')

parser.add_argument('-ready', '--ready', action='store_true')

parser.add_argument('-samples', '--samples', default=0, type=int)
parser.add_argument('-cols', '--cols', default=5, type=int)
parser.add_argument('-show_error', '--show_error', action='store_true')
parser.add_argument('-seed', '--seed', type=int, default=0)

parser.add_argument('-anim', '--anim', action='store_true')
parser.add_argument('-anim_interval', '--anim_interval', type=int, default=100)

parser.add_argument('-ES', '--energy_spectrum', action='store_true')

args = parser.parse_args()

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
    data_array = np.random.randn(2,3,64,64).astype(np.float32)


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

AE_name = f'AE_bench{benchmark}_down{args.down}_dim{args.embed_dim}'
name = f'bench{benchmark}_dt{args.dts}_r{args.rollout}_model{args.model if not args.dummy else "_dummy"}'

model = ModelClass(name)

#model.load_state_dict(['model', 'train_loss', 'val_loss'])
#model.load_state_dict(['encoder', 'decoder', 'AE_train_loss', 'AE_val_loss'], model_name = AE_name)
model.load_state_dict()
model.AE_test_mode()
model.models_test_mode()

if args.curves:
    plot_learning_curves(model.AE_train_loss, model.AE_val_loss, show=False,
                         save=f'{model.result_dir}/AE_learning_curves')
    for dt in args.dts.split(','):
        dt = int(dt)
        plot_learning_curves(model.train_losses[dt], model.val_losses[dt], show=False,
                             save=f'{model.result_dir}/dt({dt})_learning_curves')

if (not args.ready and args.final) or args.samples:
    model.set_datasets(dataset2d(data_array, train_idx),
                       dataset2d(data_array, val_idx))

if args.final:
    model.plot_final_results(ylim=args.ylim, title=f'NS{args.benchmark} R{args.rollout}')

if args.samples or args.anim or args.points or args.energy_spectrum:
    np.random.seed(args.seed)
    idxs = np.random.choice(model.data['val'].N, size=args.samples, replace=False)
    trues, preds = [], []
    for idx in idxs:
        pred, true = model.test_rollout('val', data_idx=idx, dts=args.use_dts,
                                        t_start = args.start_end[0],
                                        t_end = args.start_end[1],
                                        overtime=args.overtime)
        preds.append(pred)
        trues.append(true)
    preds = np.stack(preds)
    trues = np.stack(trues)
    if args.samples<=5:
        plot_result_samples(preds, trues, show=False, t0=args.start_end[0],
                            cols=args.cols, show_error=args.show_error,
                            save=model.result_dir+f'/seed{args.seed}_t{args.start_end}{"_overtime" if args.overtime else ""}.png',
                            overtime=args.overtime)
    
        if args.point:
            plot_result_samples_point(preds, trues, show=False, t0=args.start_end[0],
                                    save=model.result_dir+f'/seed{args.seed}_t{args.start_end}_point.png')

        if args.anim:
            for i, idx in enumerate(idxs):
                viz_compare(preds[i], trues[i], cmap='seismic', figsize=(12,4),
                            shading='gouraud', vmin=-1, vmax=1, interval=args.anim_interval,
                            dt=1, show=False, t0=args.start_end[0]+1,
                            save=model.result_dir+f'/val{idx}_t{args.start_end[0]+1}_dts{args.use_dts}.mp4')
            
    if args.energy_spectrum:
        compare_ES(preds, trues, save=model.result_dir+f'/seed{args.seed}_t{args.start_end}_ES.png',
                   title=f'timesteps {args.start_end[0]}-{args.start_end[1]}')


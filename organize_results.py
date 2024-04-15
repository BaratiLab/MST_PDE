#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 22:40:18 2023

@author: cmu
"""
import os
import shutil
import pickle
import matplotlib.pyplot as plt
from ModelClass import ModelClass
import numpy as np
import matplotlib
matplotlib.rcParams['figure.autolayout'] = True

images_dir = 'final_images'

if not os.path.exists(images_dir):
    os.mkdir(images_dir)

models = [m for m in os.listdir('Results') if m.startswith('bench')]
#model_lists = ['bench1_dt1,2,4,8_r1_model1']

for model in models:
    src = 'Results/' + model + '/final_plot.png'
    dst = images_dir + '/' + model + '.png'
    try:
        shutil.copy(src, dst)
    except:
        pass

def sorter(model_name):
    rollout = int(model_name.split('_')[-2][1:])
    model_idx = int(model_name.split('_')[-1][5:])
    return (rollout, model_idx)

# bar plot setting
colors = ['red', 'orange', 'lightgreen', 'blue']
plt.rcParams.update({'font.size': 20})
w = 0.05

for b in [1,2,3]:
    models = [m for m in os.listdir('Results') if m.startswith(f'bench{b}')]
    models = sorted(models, key=sorter)
    # print('\n'.join(models))

    string = f'bench {b}\n'
    string += 80*'='+'\n'

    results = {m:{r:{n_dts:[] for n_dts in [1,2,3,4]} for r in [1,2,4,8]} for m in range(6)}

    plt.figure(figsize=(20,5))

    for i, model in enumerate(models):
        Model = ModelClass(model)
        with open(Model.result_dir+'/final_results.pickle', 'rb') as f:
            result_dict = pickle.load(f)

        r, model_idx = sorter(model)

        for n_dts in [1,2,3,4]:
            model_type = (model_idx-1)//3
            results[model_type][r][n_dts].append(result_dict[n_dts]['val_loss'].mean())

    for model_type in range(5):
        for ir,r in enumerate([1,2,4,8]):
            if r==8 and b==1: continue
            for idt, n_dts in enumerate([1,2,3,4]):
                this_results = 100*np.array(results[model_type][r][n_dts])
                string += f'model {model_type} | rollout {r} | n_dts : {n_dts}\n'
                string += f'{np.mean(this_results):.2f} +- {np.std(this_results):.2f}\n'
                string += f'best {np.min(this_results):.2f}\n'

                plt.bar(model_type+(4*idt+ir-7.5)*w, np.mean(this_results), 0.5*w,
                        color=colors[idt], yerr=np.std(this_results))

            string += '\n'
        string += 80*'='+'\n'

    plt.xticks(np.arange(5), [f'1S    2S    3S    4S\nM{i}' for i in range(5)])
    plt.ylabel("nRMSE (%)")
    ylim = {1:[0,4], 2:[10,20], 3:[10,20]}
    plt.xlim(-0.5, 4.5)
    plt.ylim(ylim[b])
    plt.grid(linestyle='--', axis='y')
    plt.yticks(np.linspace(*ylim[b], 11))
    plt.title(f'The effect of multi-scale time-stepping and rollout on NS{b} dataset')
    plt.savefig(f'bench{b}.png')
    plt.close()

    with open(f'bench{b}.txt', 'w') as f:
            f.write(string)
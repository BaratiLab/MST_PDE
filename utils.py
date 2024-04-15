# -*- coding: utf-8 -*-
"""
@author: AmirPouya Hemmasian (ahemmasi@andrew.cmu.edu)
"""
import numpy as np
from matplotlib import animation, pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torch
import os
from Model import L2normLoss
import matplotlib
matplotlib.rcParams['figure.autolayout'] = True

DATA_DIR = '/media/pouya/DATA/PDE_data/FNO_data'

#plt.rcParams['figure.figsize'] = [8, 6]
#plt.rcParams.update({'font.size': 20})

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def what_is(x):
    print('It is', type(x))
    try:
        print('dtype is', x.dtype)
    except:
        pass
    try:
        print('shape is', x.shape)
    except:
        pass


def load_data(data='NS_V1e-5_N1200_T20', tensor=True):
    """
    Parameters
    ----------
    data : str
        the name of the .npy file containing the data

    Returns
    -------
    torch float tensor of shape (N, T, H, W)
        data array as a torch tensor
    """
    array = np.load(DATA_DIR+'/'+data+'.npy').astype(np.float32)
    if not tensor:
        return array
    return torch.as_tensor(array, dtype=torch.float)


def count_params(model):
    return sum([p.numel() for p in model.parameters()])

# %% Visualization

def viz_video(x, cmap='seismic', figsize=(8, 6), interval=10, vmin=-1, vmax=1,
              shading='gouraud', save='', show=True):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(x[0], cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    fig.colorbar(im)

    def animate(i):
        im.set_array(x[i])
        ax.set_xlabel(f't = {i}')
        return im
    clip = animation.FuncAnimation(fig, animate, frames=len(x),
                                   interval=interval, repeat=False)
    if save:
        clip.save(save)
    if show:
        plt.show()
    return clip


def viz_compare(pred, true, cmap='seismic', figsize=(12,4), shading='gouraud',
                vmin=-1, vmax=1, interval=100, dt=1, save='', show=True, t0=9,
                font=20):
    """
    Plots animation of prediction and ground truth over time
    """
    error = L2normLoss(pred, true, dim=(-1,-2), mean=False)
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=figsize)
    im0 = ax[0].pcolormesh(true[0], cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    im1 = ax[1].pcolormesh(pred[0], cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    im2 = ax[2].pcolormesh(pred[0]-true[0], cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax[0].set_title('True', fontsize=font)
    ax[1].set_title('Pred', fontsize=font)
    ax[2].set_title('Error', fontsize=font)
    ax[1].set_xlabel(f't={t0+1}', fontsize=font)
    ax[2].set_xlabel(f'error={100*error[0]:.2f}%', fontsize=font)

    for axis in ax:
        axis.set_xticks([])
        axis.set_yticks([])

    def animate(i):
        im0.set_array(true[i])
        im1.set_array(pred[i])
        im2.set_array(pred[i]-true[i])
        ax[1].set_xlabel(f't={t0+i*dt+1}', fontsize=font)
        ax[2].set_xlabel(f'error={100*error[i]:.2f}%', fontsize=font)

    anim = animation.FuncAnimation(fig, animate, frames=len(true),
                                   interval=interval, blit=False, repeat=False)
    if save:
        anim.save(save)
    if show:
        plt.show()
    return anim


def plot_result_samples(preds, trues, cols=5, t0=10, show_error=False, font=20,
                        cmap='seismic', vmin=-1, vmax=1, shading='gouraud',
                        save='', show=True, figsize=(10,10), overtime=False):
    if not overtime:
        assert preds.shape == trues.shape
    N, T, H, W = preds.shape
    fig = plt.figure(figsize=figsize)
    sr = 2 + show_error
    grid = ImageGrid(fig, 111, nrows_ncols=(sr*N, cols),
                     axes_pad=0.01,
                     share_all=True,
                     cbar_location='right',
                     cbar_mode='single',
                     cbar_size='3%',
                     cbar_pad='2%')
    ts = np.arange(T//cols-1, T, T//cols)
    for i, ax in enumerate(grid):
        row = i//cols
        col = i%cols
        idx = row//sr
        t = ts[col]
        if row == 0:
            ax.set_title(f't={t0+t+1}', fontsize=font)
        if row % sr == 0:
            try:
                ax.pcolormesh(trues[idx][t], cmap=cmap,
                              vmin=vmin, vmax=vmax, shading=shading)
            except:
                pass
            if col == 0:
                ax.set_ylabel('True', fontsize=font)
        elif row % sr == 1:
            cm = ax.pcolormesh(preds[idx][t], cmap=cmap,
                          vmin=vmin, vmax=vmax, shading=shading)
            if col == 0:
                ax.set_ylabel('Pred', fontsize=font)
        else:
            try:
                ax.pcolormesh(preds[idx][t]-trues[idx][t], cmap=cmap,
                              vmin=vmin, vmax=vmax, shading=shading)
            except:
                pass
            if col == 0:
                ax.set_ylabel('Error', fontsize=font)

        ax.set_xticks([])
        ax.set_yticks([])
    cbar = plt.colorbar(cm, cax=grid.cbar_axes[0])
    cbar.ax.tick_params(labelsize=font)
    if save:
        plt.savefig(save)
    if show:
        plt.show()


def plot_result_samples_point(preds, trues, n=0, t0=9, font=20, pos=(0.5,0.5), save='', show=True):
    N, T, H, W = preds.shape
    ix, iy = int(pos[0]*H), int(pos[1]*W)
    fig, axes = plt.subplots(nrows=N, ncols=1, sharex=True, sharey=True, figsize=(6, 4*N))

    for i in range(N):
        if N==1:
            ax = axes
        else:
            ax = axes[i]
        ax.plot(np.arange(t0+1, t0+T+1), trues[i, :, ix, iy], label='True', color='black')
        ax.plot(np.arange(t0+1, t0+T+1), preds[i, :, ix, iy], label='Pred', color='blue', linestyle='--')
        ax.set_xlabel('t', fontsize=font)
        ax.set_ylabel('u', fontsize=font)
        ax.xaxis.set_tick_params(labelsize=font)
        ax.yaxis.set_tick_params(labelsize=font)
    plt.legend(fontsize=font)
    plt.grid(linestyle='--')
    if save:
        plt.savefig(save)
    if show:
        plt.show()


def plot_learning_curves(train_loss, val_loss, skip_first=0, save='', show=True, title=''):
    """
    plots the learning curves
    Parameters
    ----------
    train_loss : list
        containing the history of training loss.
    val_loss : list
        DESCRIPTION.
    skip_first : int
        The number of first epochs to exclude in the plot.

    Returns
    -------
    None.

    """
    n_epoch = len(train_loss)
    plt.figure(figsize=(6, 4))
    epochs = range(skip_first+1, n_epoch+1)
    plt.plot(epochs, train_loss[skip_first:], label='train')
    plt.plot(epochs, val_loss[skip_first:], label='val')
    plt.xlabel('epoch')
    plt.title(title)
    plt.ylabel('loss')
    plt.grid()
    plt.legend()
    if save:
        plt.savefig(save+'.png')
    if show:
        plt.show()

# Code for plotting energy spectrum ####################################################
# This code may not be general and is a bit hard-coded

def energy_spectrum(omega):
    w_h = np.fft.rfft2(omega)
    s = omega.shape[0]
    nk = s//2 + 1
    kx = np.fft.fftfreq(s, d=1./s)
    ky = np.fft.fftfreq(s, d=1./s)
    k2 = kx[:nk]**2 + ky[:,np.newaxis]**2
    k2I = np.zeros((s, nk), dtype='complex128')
    fk = k2 != 0.0
    k2I[fk]  = 1./k2[fk]
    psih = w_h * k2I
    # angle averaged TKE spectrum
    res = 128  # num of bins
    tke = np.real(.5*k2*psih*np.conj(psih))
    kmod = np.sqrt(k2)
    k = np.arange(1, nk, 1, dtype=np.float64) # nyquist limit for this grid
    E = np.zeros_like(k)
    dk = (np.max(k)-np.min(k))/res

    #  binning energies with wavenumber modulus in threshold
    for i in range(len(k)):
        E[i] += np.sum(tke[(kmod<k[i]+dk) & (kmod>=k[i]-dk)])
    
    E /= np.sum(E)
    return E, k


def compare_ES(pred, true, save='', title=''):
    N, T = pred.shape[:2]
    E_pred = np.zeros((N, T, 32))
    E_true = np.zeros((N, T, 32))
    for i in range(N):
        for t in range(T):
            e_pred, k  = energy_spectrum(pred[i,t])
            e_true, k = energy_spectrum(true[i,t])
            E_pred[i,t] = e_pred
            E_true[i,t] = e_true
    
    E_pred_mean = np.mean(E_pred, axis=(0, 1))
    E_true_mean = np.mean(E_true, axis=(0, 1))

    plt.figure(figsize=(5,3))
    ax = plt.gca()
    plt.loglog(k, E_pred_mean, label="Pred")
    plt.loglog(k, E_true_mean, label="True")
    plt.xlabel(r'$k$', fontsize=16)
    plt.ylabel(r'$E(k)$', fontsize=16)
    plt.legend()
    #ax.set_xticks(10**(np.linspace(0, 2, 10)), minor=True)
    #plt.xlim(left=1e0, right=1e2)
    plt.grid(linestyle='--')
    # And a corresponding grid
    #ax.grid(which='both', linestyle='-.')
    plt.ylim(bottom=1e-12, top=1e0)
    plt.title(title)
    plt.savefig(save)
    #plt.show()
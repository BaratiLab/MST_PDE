# -*- coding: utf-8 -*-
"""
@author: AmirPouya Hemmasian (ahemmasi@andrew.cmu.edu)
"""
import pickle
import os
from tqdm import tqdm
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from Model import L2normLoss
import matplotlib
import matplotlib.pyplot as plt
import time
matplotlib.rcParams['figure.autolayout'] = True
plt.rcParams['figure.figsize'] = [8, 5]
#plt.rcParams.update({'font.size': 16})
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelClass():

    def __init__(self, name='model', first=False):
        self.name = name
        self.result_dir = 'Results/'+name
        if first and not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

        self.models = nn.ModuleDict()
        self.train_losses = {}
        self.val_losses = {}
        self.data = {}

    # %% Setting the Training and Validation Datasets
    def set_datasets(self, train_dataset, val_dataset):
        self.data['train'] = train_dataset
        self.data['val'] = val_dataset

    # %% Setting the main model, and its optimizer and learning curves
    def set_AE(self,
               encoder,
               decoder,
               device = Device
               ):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.AE = nn.Sequential(self.encoder, self.decoder).to(device)
        self.AE_train_loss = []
        self.AE_val_loss = []

    def set_model(self,
                  model,
                  dt = 1,
                  device = Device
                  ):
        self.models[str(dt)] = model.to(device)
        self.train_losses[dt] = []
        self.val_losses[dt] = []

    # %% Saving and Loading the state_dict
    def save_state_dict(self,
                        attrs = None,
                        path = None,
                        overwrite = False,
                        verbose = False
                        ):
        attrs = attrs or ['encoder', 'decoder', 'AE_train_loss', 'AE_val_loss',
                          'models', 'train_losses', 'val_losses', 'model_config']

        if overwrite:
            try:
                state_dict = self.load_state_dict(assign=False)
            except:
                state_dict = {}
        else:
            state_dict = {}

        for attr in attrs:
            if attr in vars(self):
                state_dict[attr] = getattr(self, attr)

        path = path or self.result_dir+'/state_dict.pickle'
        with open(path, 'wb') as f:
            pickle.dump(state_dict, f)
            if verbose:
                print('saved', list(state_dict.keys()))

    def load_state_dict(self,
                        attrs = None,
                        model_name = None,
                        assign = True,
                        verbose = False
                        ):

        if model_name is None:
            path = self.result_dir + '/state_dict.pickle'
        else:
            path = 'Results/' + model_name + '/state_dict.pickle'

        with open(path, 'rb') as f:
            state_dict = pickle.load(f)

        if attrs is not None:
            state_dict = {attr: state_dict[attr] for attr in attrs if attr in state_dict}

        if assign:
            for key, value in state_dict.items():
                setattr(self, key, value)
            if verbose:
                print('loaded', list(state_dict.keys()))

        return state_dict

# %% AutoEncoder
    def train_AE(self,
                 criterion = L2normLoss,
                 epochs = 50,
                 batch_size = 64,
                 optimizer = optim.Adam,
                 opt_kwargs = {},
                 device = Device,
                 use_cache=True
                 ):

        self.data['train'].config()
        self.data['val'].config()

        self.AE_test_mode(device, test=False)

        opt = optimizer(self.AE.parameters(), **opt_kwargs)

        train_loader = DataLoader(self.data['train'], batch_size=batch_size,
                                  shuffle=True)
        val_loader = DataLoader(self.data['val'], batch_size=batch_size,
                                shuffle=False)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                         factor=0.2,
                                                         patience=5)
        train_timer_traindata = np.zeros((epochs, len(train_loader)))
        test_timer_traindata = np.zeros((epochs, len(train_loader)))
        test_timer_testdata = np.zeros((epochs, len(val_loader)))
        for epoch in range(epochs):
            print(f'Epoch {epoch+1:03d}/{epochs:03d} , AutoEncoder of {self.name}')
            ###################################################################
            # Training:
            self.AE.requires_grad_().train()
            for i, (x, ) in enumerate(tqdm(train_loader, position=0, leave=True)):
                start_time = time.time()
                opt.zero_grad()
                x = x.to(device)
                x_rec = self.AE(x)
                loss = criterion(x_rec, x)
                loss.backward()
                # nn.utils.clip_grad_norm_(self.AE.parameters(), 1)
                opt.step()
                if not use_cache:
                    torch.cuda.empty_cache()
                torch.cuda.synchronize()
                end_time = time.time()
                this_time = end_time-start_time
                train_timer_traindata[epoch, i] = this_time
            ###################################################################
            # Validation:
            self.AE.requires_grad_(False).eval()

            sum_loss = 0.
            sum_n = 0
            for i, (x, ) in enumerate(tqdm(train_loader, position=0, leave=True)):
                start_time = time.time()
                n = x.shape[0]
                x = x.to(device)
                x_rec = self.AE(x)
                loss = criterion(x_rec, x)
                sum_loss += loss.item()*n
                sum_n += n
                if not use_cache:
                    torch.cuda.empty_cache()
                torch.cuda.synchronize()
                end_time = time.time()
                this_time = end_time-start_time
                test_timer_traindata[epoch, i] = this_time
            train_loss = sum_loss/sum_n
            self.AE_train_loss.append(train_loss)
            scheduler.step(train_loss)

            sum_loss = 0.
            sum_n = 0
            for i, (x, ) in enumerate(tqdm(val_loader, position=0, leave=True)):
                start_time = time.time()
                n = x.shape[0]
                x = x.to(device)
                x_rec = self.AE(x)
                loss = criterion(x_rec, x)
                sum_loss += loss.item()*n
                sum_n += n
                if not use_cache:
                    torch.cuda.empty_cache()
                torch.cuda.synchronize()
                end_time = time.time()
                this_time = end_time-start_time
                test_timer_testdata[epoch, i] = this_time
            val_loss = sum_loss/sum_n
            self.AE_val_loss.append(val_loss)
            ###################################################################

            if train_loss == min(self.AE_train_loss):
                self.save_state_dict()
            else:
                self.save_state_dict(['AE_train_loss', 'AE_val_loss'],
                                     overwrite=True)
            ###################################################################
            # Printing:
            print(f'Train Loss: {train_loss:.6f}')
            print(f'  Val Loss: {val_loss:.6f}')
            print(100*'-')

        # Skipping the first epoch so the data has been cached to cuda
        train_timer_traindata = train_timer_traindata[1:, :]
        test_timer_traindata = test_timer_traindata[1:, :]
        test_timer_testdata = test_timer_testdata[1:, :]
        print(f'Train time per iteration for train data:')
        print(f'{train_timer_traindata.mean():.6f}\u00B1{train_timer_traindata.std():.6f} seconds')
        print(f'Test time per iteration for train data:')
        print(f'{test_timer_traindata.mean():.6f}\u00B1{test_timer_traindata.std():.6f} seconds')
        print(f'Test time per iteration for test data:')
        print(f'{test_timer_testdata.mean():.6f}\u00B1{test_timer_testdata.std():.6f} seconds')
        print(100*'=')

    def AE_test_mode(self, device=Device, test=True):
        if test:
            self.encoder.requires_grad_(False).eval().to(device)
            self.decoder.requires_grad_(False).eval().to(device)
            self.AE = nn.Sequential(self.encoder, self.decoder)
            self.AE.requires_grad_(False).eval().to(device)
        else:
            self.encoder.requires_grad_().train().to(device)
            self.decoder.requires_grad_().train().to(device)
            self.AE = nn.Sequential(self.encoder, self.decoder)
            self.AE.requires_grad_().train().to(device)

    def AE_test_single(self, x, device=Device):
        return self.AE(x.to(device).unsqueeze(0)).squeeze(0)

    # %% The dynamical model

    def train(self,
              dt = 1,
              rollout = 1,
              criterion = L2normLoss,
              epochs = 50,
              batch_size = 64,
              optimizer = optim.Adam,
              opt_kwargs = {},
              device = Device,
              use_cache=True
              ):

        rollout = self.data['train'].config(dt=dt, rollout=rollout)
        self.data['val'].config(dt=dt, rollout=rollout)

        model = self.models[str(dt)]
        train_losses, val_losses = [], []

        model.to(device)
        self.AE_test_mode(device)

        opt = optimizer(model.parameters(), **opt_kwargs)

        train_loader = DataLoader(self.data['train'], batch_size=batch_size,
                                  shuffle=True)
        val_loader = DataLoader(self.data['val'], batch_size=batch_size,
                                shuffle=False)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                         factor=0.2,
                                                         patience=5)
        train_timer_traindata = np.zeros((epochs, len(train_loader)))
        test_timer_traindata = np.zeros((epochs, len(train_loader)))
        test_timer_testdata = np.zeros((epochs, len(val_loader)))
        for epoch in range(epochs):
            print(f'Epoch {epoch+1:03d}/{epochs:03d} , dynamical model of {self.name} for dt={dt}')
            ###################################################################
            # Training:
            model.requires_grad_().train()

            for i, xs in enumerate(tqdm(train_loader, position=0, leave=True)):
                start_time = time.time()
                opt.zero_grad()
                xs = [self.encoder(x.to(device)) for x in xs]
                latent = xs[0]
                loss = 0.
                for t in range(1, rollout+1):
                    latent = model(latent)
                    loss = loss + criterion(latent, xs[t])
                loss = loss / rollout
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                opt.step()
                if not use_cache:
                    torch.cuda.empty_cache()
                torch.cuda.synchronize()
                end_time = time.time()
                this_time = end_time-start_time
                train_timer_traindata[epoch, i] = this_time
            ###################################################################
            # Validation:
            model.requires_grad_(False).eval()

            sum_loss = 0.
            sum_n = 0
            for i, xs in enumerate(tqdm(train_loader, position=0, leave=True)):
                start_time = time.time()
                xs = [self.encoder(x.to(device)) for x in xs]
                latent = xs[0]
                loss = 0.
                for t in range(1, rollout+1):
                    latent = model(latent)
                    loss = loss + criterion(latent, xs[t])
                loss = loss / rollout
                sum_loss += loss.item()
                sum_n += latent.shape[0]
                if not use_cache:
                    torch.cuda.empty_cache()
                torch.cuda.synchronize()
                end_time = time.time()
                this_time = end_time-start_time
                test_timer_traindata[epoch, i] = this_time
                    
            train_losses += [sum_loss/sum_n]
            scheduler.step(train_losses[-1])

            sum_loss = 0.
            sum_n = 0
            for i, xs in enumerate(tqdm(val_loader, position=0, leave=True)):
                start_time = time.time()
                xs = [self.encoder(x.to(device)) for x in xs]
                latent = xs[0]
                loss = 0.
                for t in range(1, rollout+1):
                    latent = model(latent)
                    loss = loss + criterion(latent, xs[t])
                loss = loss / rollout
                sum_loss += loss.item()
                sum_n += latent.shape[0]
                if not use_cache:
                    torch.cuda.empty_cache()
                torch.cuda.synchronize()
                end_time = time.time()
                this_time = end_time-start_time
                test_timer_testdata[epoch, i] = this_time
            val_losses += [sum_loss/sum_n]
            ###################################################################
            # saving:
            if train_losses[-1] == min(train_losses):
                self.models[str(dt)] = model
            self.train_losses[dt] = train_losses
            self.val_losses[dt] = val_losses
            self.save_state_dict()
            ###################################################################
            # printing:
            print(f'Train Loss: {train_losses[-1]:.6f}')
            print(f'  Val Loss: {val_losses[-1]:.6f}')
            print(100*'-')
            if (train_losses[-1] is np.nan) or train_losses[-1]>10:
                print('Loss is nan or too big. Quitting now ...')
                break

        self.data['train'].config()
        self.data['val'].config()

        # Skipping the first epoch so the data has been cached to cuda
        train_timer_traindata = train_timer_traindata[1:, :]
        test_timer_traindata = test_timer_traindata[1:, :]
        test_timer_testdata = test_timer_testdata[1:, :]
        print(f'Train time per iteration for train data:')
        print(f'{train_timer_traindata.mean():.6f}\u00B1{train_timer_traindata.std():.6f} seconds')
        print(f'Test time per iteration for train data:')
        print(f'{test_timer_traindata.mean():.6f}\u00B1{test_timer_traindata.std():.6f} seconds')
        print(f'Test time per iteration for test data:')
        print(f'{test_timer_testdata.mean():.6f}\u00B1{test_timer_testdata.std():.6f} seconds')
        print(100*'=')

    def models_test_mode(self, device=Device, test=True):
        if test:
            self.models.requires_grad_(False).eval().to(device)
        else:
            self.models.requires_grad_().train().to(device)

    def model_test_single(self, x, dt=1, device=Device):
        encoded = self.encoder(x.to(device).unsqueeze(0))
        encoded_next = self.models[str(dt)](encoded)
        decoded_next = self.decoder(encoded_next).squeeze(0)
        return decoded_next

    # %% Using the model Autoregressively

    def test_single(self,
                    dataset = 'val',
                    data_idx = 0,
                    t_start = 9,
                    t_end = None,
                    dts = None,
                    device = Device
                    ):
        T = self.data[dataset].T
        self.data[dataset].config()
        if t_end is None or t_end>=T:
            t_end = T - 1

        x_start = self.data[dataset].get_item(data_idx, t_start)
        latent = self.encoder(x_start.unsqueeze(0).to(device))

        dts = dts or [int(dt) for dt in self.models.keys()]
        assert all([str(dt) in self.models.keys() for dt in dts]), 'missing a model for some of the specified dt'

        steps = []
        Dt = t_end - t_start
        for dt in sorted(dts, reverse=True):
            for i in range(Dt//dt):
                steps.append(dt)
                latent = self.models[str(dt)](latent)
            Dt = Dt%int(dt)
            if Dt==0:
                break

        assert sum(steps) == Dt

        x_end_pred = self.decoder(latent).squeeze(0).cpu()
        x_end = self.data[dataset].get_item(data_idx, t_end)

        return x_end_pred.numpy(), x_end.numpy(), steps

    def test_rollout(self,
                 dataset = 'val',
                 data_idx = 0,
                 t_start = 9,
                 t_end = None,
                 dts = None,
                 device = Device,
                 overtime = False
                 ):
        T = self.data[dataset].T
        self.data[dataset].config()
        t_end_pred = t_end
        if t_end is None or t_end>=T:
            t_end = T - 1

        if not overtime:
            t_end_pred = t_end

        trues = [self.data[dataset].get_item(data_idx, t) for t in range(t_start, t_end+1)]

        x_start = trues[0]
        latent0 = self.encoder(x_start.unsqueeze(0).to(device))
        Dt = t_end_pred - t_start

        dts = dts or [int(dt) for dt in self.models.keys()]
        assert all([str(dt) in self.models.keys() for dt in dts]), 'missing model for some of the specified dt'

        preds = [latent0] + Dt*[None]
        prev_dt = Dt + 1
        preds_idxs = [0]
        for dt in sorted(dts, reverse=True):
            new_preds_idxs = []
            for t in preds_idxs:
                latent = preds[t]
                for i in range(t+dt, min(t+prev_dt, Dt+1), dt):
                    if preds[i] is not None:
                        break
                    latent = self.models[str(dt)](latent)
                    preds[i] = latent
                    new_preds_idxs.append(i)
            preds_idxs += new_preds_idxs
            prev_dt = dt

        preds_idxs = sorted(preds_idxs)

        assert preds_idxs == list(range(Dt+1))
        assert all([x is not None for x in preds])

        preds = [self.decoder(x).squeeze(0).cpu() for x in preds]
        # excluding  the starting time step:
        preds = torch.cat(preds[1:])
        trues = torch.cat(trues[1:])
        return preds, trues

    def get_final_scores(self, dts=None):
        self.AE_test_mode()
        self.models_test_mode()
        results_dict = {}
        dts = dts or [int(dt) for dt in self.models.keys()]

        for it in range(1, len(dts)+1):

            train_loss = []
            val_loss = []
            train_loss_timed = []
            val_loss_timed = []

            print('with dts', dts[:it])

            for i in tqdm(range(self.data['train'].N), position=0, leave=True):
                pred, true = self.test_rollout('train', i, dts=dts[:it])
                train_loss.append(
                    L2normLoss(pred, true, dim=(-1,-2,-3), mean=False))
                train_loss_timed.append(
                    L2normLoss(pred, true, dim=(-1,-2), mean=False))

            for i in tqdm(range(self.data['val'].N), position=0, leave=True):
                pred, true = self.test_rollout('train', i, dts=dts[:it])
                val_loss.append(
                    L2normLoss(pred, true, dim=(-1,-2,-3), mean=False))
                val_loss_timed.append(
                    L2normLoss(pred, true, dim=(-1,-2), mean=False))

            results_dict[it] = {
                'train_loss': torch.stack(train_loss).cpu().numpy(),
                'val_loss': torch.stack(val_loss).cpu().numpy(),
                'train_loss_timed': torch.stack(train_loss_timed).cpu().numpy(),
                'val_loss_timed': torch.stack(val_loss_timed).cpu().numpy()
                }

        with open(self.result_dir+'/final_results.pickle', 'wb') as f:
            pickle.dump(results_dict, f)

        return results_dict


    def plot_final_results(self, figsize=(8,6), ylim=[0,30], title=''):
        print('Getting final scores of', self.name)
        try:
            with open(self.result_dir+'/final_results.pickle', 'rb') as f:
                results_dict = pickle.load(f)
        except:
            print('Could not load result_dict. running get_final_scores!')
            results_dict = self.get_final_scores()

        plt.figure(figsize=figsize)
        colors = {1:'red', 2:'orange', 3:'lightgreen', 4:'blue', 5:'purple'}

        T = results_dict[1]['train_loss_timed'].shape[1]
        trange = range(11, 11+T)
        for i, result_dict in results_dict.items():
            plt.plot(trange, 100*result_dict['train_loss_timed'].mean(0), color=colors[i],
                     label=f"{i}S (train)", linestyle='--')
                     #label=f"{i}S (train), total {100*result_dict['train_loss'].mean():.2f}%", linestyle='--')
            plt.plot(trange, 100*result_dict['val_loss_timed'].mean(0), color=colors[i],
                     label=f"{i}S (val)")
                     #label=f"{i}S (val)  , total {100*result_dict['val_loss'].mean():.2f}%")

        plt.legend(loc='upper left')
        plt.grid(linestyle='--')
        plt.xlabel('time')
        plt.xticks(np.linspace(10, 10+T, 6))
        plt.xlim([10,10+T])
        plt.ylabel('nRMSE (%)')
        plt.ylim(ylim)
        plt.title(title)
        plt.savefig(self.result_dir+'/final_plot.png')
        plt.close()

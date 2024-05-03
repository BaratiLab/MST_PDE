"""
This code cleans the Burgers and NavierStokes datasets from the FNO paper and saves them as .npy files.
(The KFvorticity ones are already in .npy format)

The original datasets are available in this google drive:

https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-?usp=drive_link

First, you need to download them and extract them into the `FNO_data_path`.
Then, you can run this code to clean the data and save them in `FNO_data_cleaned_path`.

Each dataset will be saved as a .npy file with shape (N, t+1, ...)
Where N is the number of samples, t is the number of timesteps, and ... is the spatial dimensions.

Usually in the original data, there is an input (initial condition) denoted as "a" or "input"
and an output (the solution at each timestep) for t timesteps, denoted as "u" or "output".
The cleaned data concatenates them together to have t+1 timesteps.

author: AmirPouya Hemmasian
a.pouyahemmasian@gmail.com
ahemmasi@andrew.cmu.edu
"""

import numpy as np
import scipy.io as sio
import h5py
from utils import what_is, file_size
import os


FNO_data_path = '/media/pouya/DATA/PDE_data/FNO_data'
FNO_data_cleaned_path = '/media/pouya/DATA/PDE_data/FNO_data_cleaned'
os.makedirs(FNO_data_cleaned_path, exist_ok=True)

def load_mat(file_path, verbose=True):

    if verbose:
        print('Getting data from:')
        print(file_path, f'({file_size(file_path)})')
        print(30*'=')

    try:
        data_dict = {
            name: np.array(obj, dtype=np.float32, copy=False)
            for name, obj in sio.loadmat(file_path).items()
            if isinstance(obj, np.ndarray) and (obj.dtype == np.float64 or obj.dtype == np.float32)
            }
        method = 'scipy.io.loadmat'
    except:
        f = h5py.File(file_path, 'r')
        data_dict = {
            name: np.array(obj, dtype=np.float32, copy=False) 
            for name, obj in f.items() 
            if isinstance(obj, h5py.Dataset)
            }
        f.close()
        method = 'h5py.File'
            
    if verbose:
        print('Loaded data from the .mat file with', method)
        print('The content of the file is:')
        for key, item in data_dict.items():
            print(30*'-')
            print(key)
            what_is(item)
        print(30*'=')

    return data_dict


# BURGERS #######################################################################

def clean_Burgers_R10(just_checking=False): # tested

    original_file_path = FNO_data_path + '/Burgers_R10.mat'
    
    data = load_mat(original_file_path, verbose=True)

    if just_checking:
        return
    
    data['u'] = np.stack([data['a'], data['u']], axis=1) # (2048, 2, 8192)

    new_file_path = FNO_data_cleaned_path + '/Burgers_R10_N2048_T1_x8192.npy'
    np.save(new_file_path, data['u'])
    print('Data cleaned and saved to:')
    print(new_file_path, f'({file_size(new_file_path)})')
    what_is(data['u'])
    print(30*'=')

    return new_file_path


def clean_Burgers_v1000_t200_r1024_N2048(just_checking=False): # tested

    original_file_path = FNO_data_path + '/Burgers_v1000_t200_r1024_N2048.mat'
    
    data = load_mat(original_file_path, verbose=True)

    if just_checking:
        return

    data['output'][:, 0, :] = data['input']
    data['output'] = np.transpose(data['output'], axes=(2, 1, 0)) # (2048, 201, 1024)

    new_file_path = FNO_data_cleaned_path + '/Burgers_v1000_N2048_t200_x1024.npy'
    np.save(new_file_path, data['output'])
    print('Data cleaned and saved to:')
    print(new_file_path, f'({file_size(new_file_path)})')
    what_is(data['output'])
    print(30*'=')

    return new_file_path


def clean_Burgers_v100_t100_r1024_N2048(just_checking=False): # tested
    original_file_path = FNO_data_path + '/Burgers_v100_t100_r1024_N2048.mat'
    
    data = load_mat(original_file_path, verbose=True)

    if just_checking:
        return
    
    data['output'][:, 0, :] = data['input']

    new_file_path = FNO_data_cleaned_path + '/Burgers_v100_N2048_t100_x1024.npy'
    np.save(new_file_path, data['output'])
    print('Data cleaned and saved to:')
    print(new_file_path, f'({file_size(new_file_path)})')
    what_is(data['output'])
    print(30*'=')

    return new_file_path

# NAVIER STOKES (NS) ##############################################################

def clean_ns_V1e_3_N5000_T50(just_checking=False): # tested

    original_file_path = FNO_data_path + '/ns_V1e-3_N5000_T50.mat'
    
    data = load_mat(original_file_path, verbose=True)

    if just_checking:
        return

    data['a'] = data['a'].transpose(2, 1, 0)[:, None, :, :] # (Y, X, N) -> (N, 1, X, Y)
    data['u'] = data['u'].transpose(3, 0, 2, 1) # (T, Y, X, N) -> (N, T, X, Y)
    data['u'] = np.concatenate([data['a'], data['u']], axis=1)
    
    new_file_path = FNO_data_cleaned_path + '/NS_v1e-3_N5000_T50.npy'
    np.save(new_file_path, data['u'])
    print('Data cleaned and saved to:')
    print(new_file_path, f'({file_size(new_file_path)})')
    what_is(data['u'])
    print(30*'=')

    return new_file_path


def clean_ns_V1e_4_N10000_T30(just_checking=False): # tested

    original_file_path = FNO_data_path + '/ns_V1e-4_N10000_T30.mat'
    
    data = load_mat(original_file_path, verbose=True)

    if just_checking:
        return

    # Despite the name of the file, the data has 50 timesteps.

    data['a'] = data['a'].transpose(2, 0, 1)[:, None, :, :] # (X, Y, N) -> (N, 1, X, Y)
    data['u'] = data['u'].transpose(3, 0, 1, 2) # (T, X, Y, N) -> (N, T, X, Y)
     
    # data['u'] = np.concatenate([data['a'], data['u']], axis=1)

    # The above code cannot succeed because the concatenated array is too large to fit in the memory.
    # So, we need to create a placeholder for the new data as a memory mapped array.
    
    # Creating a memory mapped array on the disk in the mode 'w+', and filling it with our data:
    temp_file_path = FNO_data_cleaned_path + '/temp_array.npy'
    temp_memmap = np.memmap(temp_file_path, mode='w+', dtype='float32', shape=(10000, 51, 64, 64))
    temp_memmap[:, :1, :, :] = data['a']
    temp_memmap[:, 1:, :, :] = data['u']
    # Opening the memory mapped array in the mode 'r' and saving it as a normal .npy file:
    temp_memmap = np.memmap(temp_file_path, mode='r' , dtype='float32', shape=(10000, 51, 64, 64))
    new_file_path = FNO_data_cleaned_path + '/NS_v1e-4_N10000_T50.npy'
    np.save(new_file_path, temp_memmap)
    # Removing the temporary memory mapped array:
    os.remove(temp_file_path)
    
    print('Data cleaned and saved to:')
    print(new_file_path, f'({file_size(new_file_path)})')
    what_is(temp_memmap)
    print(30*'=')

    return new_file_path
    

def clean_ns_data_V1e_4_N20_T50_R256test(just_checking=False): # tested

    original_file_path = FNO_data_path + '/ns_data_V1e-4_N20_T50_R256test.mat'
    
    data = load_mat(original_file_path, verbose=True)

    if just_checking:
        return

    data['a'] = data['a'][:, None, :, :] # (N, X, Y) -> (N, 1, X, Y)
    data['u'] = data['u'].transpose(0, 3, 1, 2) # (N, X, Y, T) -> (N, T, X, Y)
    data['u'] = data['u'][:, 3::4, :, :] # (N, 200, X, Y) -> (N, 50, X, Y)
    data['u'] = np.concatenate([data['a'], data['u']], axis=1)

    new_file_path = FNO_data_cleaned_path + '/NS_v1e-4_N20_T50_r256.npy'
    np.save(new_file_path, data['u'])
    print('Data cleaned and saved to:')
    print(new_file_path, f'({file_size(new_file_path)})')
    what_is(data['u'])
    print(30*'=')

    return new_file_path


def clean_NavierStokes_V1e_5_N1200_T20(just_checking=False): # tested

    original_file_path = FNO_data_path + '/NavierStokes_V1e-5_N1200_T20.mat'
    
    data = load_mat(original_file_path, verbose=True)

    if just_checking:
        return
    
    data['a'] = data['a'][:, None, :, :] # (N, X, Y) -> (N, 1, X, Y)
    data['u'] = data['u'].transpose(0, 3, 1, 2) # (N, X, Y, T) -> (N, T, X, Y)
    data['u'] = np.concatenate([data['a'], data['u']], axis=1)

    new_file_path = FNO_data_cleaned_path + '/NS_v1e-5_N1200_T20.npy'
    np.save(new_file_path, data['u'])
    print('Data cleaned and saved to:')
    print(new_file_path, f'({file_size(new_file_path)})')
    what_is(data['u'])
    print(30*'=')

    return new_file_path

if __name__ == '__main__':

    clean_file_path = clean_Burgers_R10()
    print('Burger_R10.mat cleaned and saved to',
          clean_file_path.split('/')[-1])
    print(50*'=')
    print()

    clean_file_path = clean_Burgers_v1000_t200_r1024_N2048()
    print('Burgers_v1000_t200_r1024_N2048.mat cleaned and saved to',
          clean_file_path.split('/')[-1])
    print(50*'=')
    print()
    
    clean_file_path = clean_Burgers_v100_t100_r1024_N2048()
    print('Burgers_v100_t100_r1024_N2048.mat cleaned and saved to ', 
          clean_file_path.split('/')[-1])
    print(50*'=')
    print()

    clean_file_path = clean_ns_V1e_3_N5000_T50()
    print('ns_V1e-3_N5000_T50.mat cleaned and saved to ', 
          clean_file_path.split('/')[-1])
    print(50*'=')
    print()
    # Saving the part used in MST_PDE paper
    np.save(FNO_data_cleaned_path + '/MST_NS_v1e-3_N1200_T50.npy',
            np.load(clean_file_path)[:1200, 1:51, :, :])

    clean_file_path = clean_ns_V1e_4_N10000_T30()
    print('ns_V1e-4_N10000_T30.mat cleaned and saved to ',
          clean_file_path.split('/')[-1])
    print(50*'=')
    print()
    # Saving the part used in MST_PDE paper
    np.save(FNO_data_cleaned_path + '/MST_NS_v1e-4_N1200_T30.npy',
            np.load(clean_file_path)[:1200, 1:31, :, :])

    clean_data_path = clean_ns_data_V1e_4_N20_T50_R256test()
    print('ns_data_V1e-4_N20_T50_R256test.mat cleaned and saved to ',
          clean_data_path.split('/')[-1])
    print(50*'=')
    print()
    
    clean_data_path = clean_NavierStokes_V1e_5_N1200_T20()
    print('NavierStokes_V1e-5_N1200_T20.mat cleaned and saved to ',
          clean_data_path.split('/')[-1])
    print(50*'=')
    print()
    # Saving the part used in the MST_PDE paper
    np.save(FNO_data_cleaned_path + '/MST_NS_v1e-5_N1200_T20.npy',
            np.load(clean_data_path)[:1200, 1:21, :, :])

    print('All data cleaned and saved successfully.')

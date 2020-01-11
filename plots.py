# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import csv, os
from glob import glob

# %%
def curve_plots(u_accus_l, a_accus_l, a_locations_l, methods_l, 
                   saveimg=False, title=''):
    
    u_bins = len(u_accus_l[0])
    u_bin_size = 1 / u_bins
    u_locations_all = np.arange(0+u_bin_size/2, 1+u_bin_size/2, u_bin_size)
    
    # remove the 0s
    u_locations_l = []
    for i in range(len(u_accus_l)):
        i_0s = np.where(u_accus_l[i] == 0.0)
        u_accus_l[i] = np.delete(u_accus_l[i], i_0s)
        u_locations_l.append(np.delete(u_locations_all, i_0s))
    
    plt.style.use('ggplot')
#    plt.style.use('bmh')
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10.5, 5), sharex='col', sharey='row')
    fig.subplots_adjust(wspace=0.15)
#    fig.suptitle(title, fontsize=19)
    
    # Uniform plot
    u_lines = [None] * len(u_accus_l)
    for i in range(len(u_accus_l)):
        u_lines[i] = ax[0].plot(u_locations_l[i], u_accus_l[i], label=methods_l[i])

    ax[0].set_aspect('equal')
    ax[0].plot([0,1], [0,1], linestyle="--", color="#52854C")
    ax[0].legend(fontsize=12, loc=2)
    ax[0].set_title('Uniform Binning', fontsize=15)
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)
    ax[0].set_xlabel('Confidence', fontsize=15, color = "black")
    ax[0].set_ylabel('Accuracy', fontsize=15, color = "black")
    
    # Adaptive plot
    a_lines = [None] * len(a_accus_l)
    for i in range(len(a_accus_l)):
        a_lines[i] = ax[1].plot(a_locations_l[i], a_accus_l[i], label=methods_l[i])

    ax[1].set_aspect('equal')
    ax[1].plot([0,1], [0,1], linestyle="--", color="#52854C")
    ax[1].legend(fontsize=12, loc=2)
    ax[1].set_title('Adaptive Binning', fontsize=15)
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel('Confidence', fontsize=15, color = "black")
#    ax[1].set_ylabel('Accuracy', fontsize=15, color = "black")
    
    if saveimg:
        dir_imgs = './logs/imgs/'
        if not os.path.exists(dir_imgs):
            os.makedirs(dir_imgs)
        plt.savefig(dir_imgs + f'{title}.png', format='png', dpi=100, transparent=True, bbox_inches='tight')
    
    return
# %%
CSV_DIR = '../results_MNIST/'
DATA_NAME = 'MNIST'
# in ['MNIST', 'OC']
ARCHI_NAME = 'ResNet'
# in ['LeNet', 'ResNet']
LABEL_SMOOTH = '*'
# between 0-1, 0.1 shoube be enough
METHOD_NAME = '*'
# in ['Base', 'Drop', 'CDrop', 'LLCDrop', 'VI', 'LLVI']
CAL_NAME = 'TS'
# in ['NC', 'TS']


pattern = f'{DATA_NAME}_{ARCHI_NAME}_LS{LABEL_SMOOTH}_{METHOD_NAME}_{CAL_NAME}.csv'

def read_results(pattern):
    method_names = []
    u_accus_l = []
    a_accus_l = []
    a_locations_l = []
    
    for csv_path in glob(CSV_DIR+pattern):
        
        title = os.path.basename(csv_path)[:-len('.csv')]
        method_name = '_'.join(title.split('_')[2:])
        method_names.append(method_name)
        
        with open(csv_path, 'r', newline='') as f:
            f_csv = csv.reader(f)
            u_accus = np.asarray(list(map(float, next(f_csv))))
            u_gaps = list(map(float, next(f_csv)))
            u_neg_gaps = list(map(float, next(f_csv)))
            a_locations = np.asarray(list(map(float, next(f_csv))))
            a_accus = np.asarray(list(map(float, next(f_csv))))
            a_gaps = list(map(float, next(f_csv)))
            a_neg_gaps = list(map(float, next(f_csv)))
            a_widths = list(map(float, next(f_csv)))
        u_accus_l.append(u_accus)
        a_accus_l.append(a_accus)
        a_locations_l.append(a_locations)
    return method_names, u_accus_l, a_accus_l, a_locations_l

method_names, u_accus_l, a_accus_l, a_locations_l = read_results(pattern)

# %%
curve_plots(u_accus_l, a_accus_l, a_locations_l, method_names, 
            saveimg=False, title=f'{DATA_NAME}_{ARCHI_NAME}')

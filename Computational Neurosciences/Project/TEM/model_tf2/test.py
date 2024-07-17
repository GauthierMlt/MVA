# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import seaborn
import datetime
import re
from os import listdir
import sys
import copy as cp
sys.path.insert(0, '../model_tf2')
import parameters
import plotting_functions as pf
import data_utils as du
import model_utils as mu
import behaviour_analyses as ba
import environments as ef
import cell_analyses as ca



path = 'C:/Users/gaumu/Desktop/TEM_linux/Summaries/'
path = '/mnt/c/Users/gaumu/Desktop/TEM_linux/Summaries/'
save_dirs = [path]

# Choose which training run data to load
date = '2024-03-11'
run = '0'
index = None  # '1000', None

# grid cells: date = [saved], '2021-10-03', run = '4', index=1000
# grid cells: date = [saved], '2021-09-05', run = '0' (hex grids, square world, relu) 
# - also use for place-grid metric figure
# grid cells: date = [saved], '2021-08-30', run = '3' (hex grids, square world, relu)
# grid cells: date = [saved], '2021-09-29', run = '7' (hex world)
# place cells: date = [saved], '2021-09-29', run = '0'

# Try to find the most recent trained model data to run a forward pass
recent = -1
time_series_smoothing = 1
try:
    # Find model path and iteration index
    save_dir, index = pf.get_model_path(run, date, save_dirs, recent, index=index)
    # Run forward path for retrieved model, if folder doesn't exist yet
    model = ba.save_trained_outputs(date, run, int(index), base_path=save_dir, \
                                    force_overwrite=False, n_envs_save=16)
except FileNotFoundError:
    print('No trained model weights found for ' + date + ', run ' + run + '.')

# Load data, generated either during training or in a forward pass through a trained model
data, para, list_of_files, save_path, env_dict = pf.get_data(save_dirs, run, date, recent, \
                                                             index=index, \
                                                             smoothing=time_series_smoothing, \
                                                             n_envs_save=16)
# Assign parameters
params, widths, n_states = para

# Specify plotting parameters. Some fields will be added after loading data & parameters
plot_specs = pf.__plot_specs

# Set plot_spec fields that depend on parameters after loading
plot_specs.index = index
plot_specs.world_type = params.world_type
plot_specs.directory = save_path.split('save/')[0]

import seaborn
#sns.set(font_scale = 2)
# seaborn.set_style(style='white')
# seaborn.set_style({'axes.spines.bottom': False,'axes.spines.left': False,'axes.spines.right': \
#                    False,'axes.spines.top': False})

masks, g_lim = pf.sort_data(data.g, widths, plot_specs)

g_reps = data.g_timeseries[0][:,:400].T
reps = model.g2p(g_reps).numpy()
n_k = reps.shape[-1]
scal_prods = np.matmul(reps, reps.T)
scal_prods = mu.threshold(scal_prods, n_k*params.kernel_thresh_min, \
                          n_k*params.kernel_thresh_max, thresh_slope=0.001).numpy()
scal_prods = scal_prods / np.sqrt(reps.shape[-1])

masks = [(np.sum(g,1) != 0).tolist() for g in data.g]
trainalbe_variables = model.trainable_weights

for env in range(min(params.n_envs_save, len(data.acc_to))):
    num_correct = np.sum(data.acc_to[env] * data.positions[env])
    proportion = num_correct / sum(data.positions[env])
    _n_states = len(data.positions[env] > 0.01)
    approx_num = proportion * _n_states
    print(env, '   Approx proportion : ', np.round(proportion, decimals=3), \
          '   Approx num : ' + str(int(np.round(approx_num, decimals=4)[0])) \
          + ' of ' + str(_n_states))
    
env0 = 2
env1 = 3
envs = [env0, env1]   

plot_specs.split_freqs = False
plot_specs.n_cells_freq = params.g_size
plot_specs.cmap = 'jet'
plot_specs.node_plot = True
plot_specs.max_min = True
plot_specs.smoothing = 0.5

pf.square_plot(data.g, env0, params, plot_specs, name='g0_'+index, lims=g_lim, mask=masks[env0], \
               env_class=env_dict.curric_env.envs[env0], fig_dir=plot_specs.directory)

pf.square_plot(data.g, env1, params, plot_specs, name='g1_'+index, lims=g_lim, mask=masks[env1], \
               env_class=env_dict.curric_env.envs[env1], fig_dir=plot_specs.directory)

gs=[]
for i, g_b in enumerate(data.g):
    gs_b = []
    for j in range(params.max_states):
        try:
            g = g_b[j]
        except IndexError:
            g = np.zeros(params.g_size)
            
        gs_b.append(g)
    gs.append(np.stack(gs_b, axis=0))
gs = np.stack(gs, axis=1)

# repeat gs to get correct batch size:
reps = np.ceil(params.batch_size/gs.shape[1]).astype(int)
gs = np.tile(gs, [1,reps,1])
gs = gs[:,:params.batch_size,:]

ps = []
g2ps = []
for g in gs:
    train_i = 1000000
    scalings = parameters.get_scaling_parameters(train_i, params)
    inputs_test_tf = mu.inputs_2_tf(env_dict.inputs, env_dict.hebb, scalings)
    memories_dict, variable_dict = model.init_input(inputs_test_tf)
    mem_step = model.mem_step(memories_dict, 0)

    retrieved_g2x, g_mem_input, hidden_g2x = model.gen_p(g, mem_step)
    
    g2ps.append(model.g2p(g).numpy())
    
    ps.append(hidden_g2x['stored']['prob'].numpy())
ps = np.stack(ps, axis=1)
g2ps = np.stack(g2ps, axis=1)


g2ps_ = []
for b, g2p in enumerate(g2ps):
    g2ps_.append(g2p[:env_dict.curric_env.envs[b].n_states,:])

pf.square_plot(g2ps_, env0, params, plot_specs, name='g2p0_'+index, lims=None, mask=masks[env0], \
               env_class=env_dict.curric_env.envs[env0], fig_dir=plot_specs.directory)

ps_ = []
for b, p in enumerate(ps):
    ps_.append(p[:env_dict.curric_env.envs[b].n_states,:])
pf.square_plot(ps_, env0, params, plot_specs, name='p0_'+index, lims=None, mask=masks[env0], \
               env_class=env_dict.curric_env.envs[env0], fig_dir=plot_specs.directory)  


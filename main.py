# -*- encoding=utf-8 -*-

import math, os, errno, pickle, time, shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# own
from dem_to_pf import *
from pf_to_dem import *
from ic import *
from tools import *
from Parameters import *

# ------------------------------------------------------------------------------------------------------------------------------------------ #
# Functions

def run_moose(dict_user, dict_sample):
    '''
    Prepare and run moose simulation.
    '''

    raise ValueError('Stop')

    # from dem to pf
    # rbm to pf variables
    tic_dem_to_pf = time.perf_counter() # compute dem_to_pf performances
    tic_tempo = time.perf_counter() # compute performances
    move_phasefield(dict_user, dict_sample) # in dem_to_pf.py
    tac_tempo = time.perf_counter() # compute performances
    dict_user['move_pf'] = dict_user['move_pf'] + tac_tempo-tic_tempo 
    # compute contact characterization
    tic_tempo = time.perf_counter() # compute performances
    compute_contact_volume(dict_user, dict_sample) # in dem_to_pf.py
    tac_tempo = time.perf_counter() # compute performances
    dict_user['comp_con_vol'] = dict_user['comp_con_vol'] + tac_tempo-tic_tempo 
    # compute diffusivity map
    tic_tempo = time.perf_counter() # compute performances
    compute_kc(dict_user, dict_sample) # in dem_to_pf.py
    tac_tempo = time.perf_counter() # compute performances
    dict_user['comp_kc'] = dict_user['comp_kc'] + tac_tempo-tic_tempo 
    # compute solid activity map
    tic_tempo = time.perf_counter() # compute performances
    compute_as(dict_user, dict_sample) # in dem_to_pf.py
    tac_tempo = time.perf_counter() # compute performances
    dict_user['comp_as'] = dict_user['comp_as'] + tac_tempo-tic_tempo 
    # compute ed (for trackers and Aitken method)
    tic_tempo = time.perf_counter() # compute performances
    compute_ed(dict_user, dict_sample) # in dem_to_pf.py
    tac_tempo = time.perf_counter() # compute performances
    dict_user['comp_ed'] = dict_user['comp_ed'] + tac_tempo-tic_tempo 
    # Aitken method
    tic_tempo = time.perf_counter() # compute performances
    compute_dt_PF_Aitken(dict_user, dict_sample) # in dem_to_pf.py
    tac_tempo = time.perf_counter() # compute performances
    dict_user['comp_dt'] = dict_user['comp_dt'] + tac_tempo-tic_tempo 
    
    # generate .i file
    tic_tempo = time.perf_counter() # compute performances
    write_i(dict_user, dict_sample) # in dem_to_pf.py
    tac_tempo = time.perf_counter() # compute performances
    tac_dem_to_pf = time.perf_counter() # compute dem_to_pf performances
    dict_user['L_t_dem_to_pf'].append(tac_dem_to_pf-tic_dem_to_pf)
    dict_user['write_i'] = dict_user['write_i'] + tac_tempo-tic_tempo 
    
    # pf
    print('Running PF')
    tic_pf = time.perf_counter() # compute pf performances
    os.system('mpiexec -n '+str(dict_user['n_proc'])+' ~/projects/moose/modules/phase_field/phase_field-opt -i pf.i')
    tac_pf = time.perf_counter() # compute pf performances
    dict_user['L_t_pf'].append(tac_pf-tic_pf)
    dict_user['solve_pf'] = dict_user['solve_pf'] + tac_pf-tic_pf 
    
    # from pf to dem
    tic_tempo = time.perf_counter() # compute performances
    last_j_str = sort_files(dict_user, dict_sample) # in pf_to_dem.py
    tac_dem_to_pf = time.perf_counter() # compute dem_to_pf performances
    tac_tempo = time.perf_counter() # compute performances
    dict_user['sort_pf'] = dict_user['sort_pf'] + tac_tempo-tic_tempo 
    
    print('Reading data')
    tic_pf_to_dem = time.perf_counter() # compute pf_to_dem performances
    read_vtk(dict_user, dict_sample, last_j_str) # in pf_to_dem.py
    tac_pf_to_dem = time.perf_counter() # compute pf_to_dem performances
    dict_user['L_t_pf_to_dem_2'].append(tac_pf_to_dem-tic_pf_to_dem)
    dict_user['read_pf'] = dict_user['read_pf'] + tac_pf_to_dem-tic_pf_to_dem 
    
# ------------------------------------------------------------------------------------------------------------------------------------------ #

def run_yade(dict_user, dict_sample):
    '''
    Prepare and run yade simulation.
    '''
    # from pf to dem
    tic_pf_to_dem = time.perf_counter() # compute pf_to_dem performances
    compute_vertices(dict_user, dict_sample) # from pf_to_dem.py
    tac_pf_to_dem = time.perf_counter() # compute pf_to_dem performances
    dict_user['L_t_pf_to_dem_1'].append(tac_pf_to_dem-tic_pf_to_dem)
    dict_user['comp_vertices'] = dict_user['comp_vertices'] + tac_pf_to_dem-tic_pf_to_dem 
    
    # shape evolution
    tic_tempo = time.perf_counter() # compute performances
    plot_shape_evolution(dict_user, dict_sample) # from tools.py
    tac_tempo = time.perf_counter() # compute performances
    dict_user['plot_shape'] = dict_user['plot_shape'] + tac_tempo-tic_tempo 

    # transmit data
    tic_tempo = time.perf_counter() # compute performances
    dict_save = {
    'E': dict_user['E'],
    'Poisson': dict_user['Poisson'],
    'force_applied': dict_user['force_applied'],
    'k_control_force': dict_user['k_control_force'],
    'd_y_limit': dict_user['d_y_limit'],
    'x_min_wall': dict_user['x_min_wall'],
    'x_max_wall': dict_user['x_max_wall'],
    'y_min_wall': dict_user['y_min_wall'],
    'y_max_wall': dict_user['y_max_wall'],
    'n_ite_max': dict_user['n_ite_max'],
    'steady_state_detection': dict_user['steady_state_detection'],
    'n_steady_state_detection': dict_user['n_steady_state_detection'],
    'print_all_dem': dict_user['print_all_dem'],
    'print_dem': 'dem' in dict_user['L_figures'],
    'i_DEMPF_ite': dict_sample['i_DEMPF_ite']
    }
    with open('data/main_to_dem.data', 'wb') as handle:
        pickle.dump(dict_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tac_tempo = time.perf_counter() # compute performances
    dict_user['save_dem'] = dict_user['save_dem'] + tac_tempo-tic_tempo 

    # dem
    print('Running DEM')
    tic_dem = time.perf_counter() # compute dem performances
    os.system('yadedaily -j '+str(dict_user['n_proc'])+' -x -n dem_base.py')
    tac_dem = time.perf_counter() # compute dem performances
    dict_user['L_t_dem'].append(tac_dem-tic_dem)
    dict_user['solve_dem'] = dict_user['solve_dem'] + tac_dem-tic_dem 
    
    # sort files
    tic_tempo = time.perf_counter() # compute performances
    sort_files_yade() # in dem_to_pf.py
    tac_tempo = time.perf_counter() # compute performances
    dict_user['sort_dem'] = dict_user['sort_dem'] + tac_tempo-tic_tempo 
    
    # load data
    tic_tempo = time.perf_counter() # compute performances
    with open('data/dem_to_main.data', 'rb') as handle:
        dict_save = pickle.load(handle)
    dict_sample['L_center'] = dict_save['L_pos']
    dict_sample['L_box'] = dict_save['L_box']
    tac_tempo = time.perf_counter() # compute performances
    dict_user['read_dem'] = dict_user['read_dem'] + tac_tempo-tic_tempo 

    # plot evolution of the number of vertices used in Yade
    tic_tempo = time.perf_counter() # compute performances
    plot_n_vertices(dict_user, dict_sample) # from tools.py
    tac_tempo = time.perf_counter() # compute performances
    dict_user['plot_n_vertices'] = dict_user['plot_n_vertices'] + tac_tempo-tic_tempo 
    
# ------------------------------------------------------------------------------------------------------------------------------------------ #
# Plan
    
# get parameters
dict_user = get_parameters() # from Parameters.py
dict_sample = {}

# folders
create_folder('vtk') # from tools.py
create_folder('plot') # from tools.py
if dict_user['print_all_contact_detection'] and 'contact_detection' in dict_user['L_figures']:
    create_folder('plot/contact_detection') # from tools.py
if dict_user['print_all_map_config'] and 'maps' in dict_user['L_figures']:
    create_folder('plot/map_etas_solute') # from tools.py
if 'shape_evolution' in dict_user['L_figures']:
    create_folder('plot/shape_evolution') # fom tools.py
if 'dem' in dict_user['L_figures'] and dict_user['print_all_dem']:
    create_folder('plot/dem') # fom tools.py
create_folder('data') # from tools.py
create_folder('input') # from tools.py
create_folder('dict') # from tools.py

# if saved check the folder does not exist
if dict_user['save_simulation']:
    # name template id k_diss_k_prec_D_solute_force_applied
    name = str(int(dict_user['k_diss']))+'_'+str(int(dict_user['k_prec']))+'_'+str(int(10*dict_user['D_solute']))+'_'+str(int(dict_user['force_applied']))
    # check
    if Path('../Data_PressureSolution_MG_2D/'+name).exists():
        raise ValueError('Simulation folder exists: please change parameters')

# compute performances
tic = time.perf_counter()

# ------------------------------------------------------------------------------------------------------------------------------------------ #
# Create initial condition

if dict_user['Shape'] == 'Sphere_no_overlap' or dict_user['Shape'] == 'Sphere_cfc':
    create_spheres(dict_user, dict_sample) # from ic.py
create_solute(dict_user, dict_sample) # from ic.py

# compute tracker
s_eta = 0
for etai in range(len(dict_sample['L_L_ig_etai'])):
    dict_user['L_L_sum_eta_i'].append([np.sum(dict_sample['L_etai_map'][etai])])
    s_eta = s_eta + np.sum(dict_sample['L_etai_map'][etai])
dict_user['L_sum_c'].append(np.sum(dict_sample['c_map']))
dict_user['L_sum_mass'].append(s_eta+np.sum(dict_sample['c_map']))
m_eta = 0
for etai in range(len(dict_sample['L_L_ig_etai'])):
    dict_user['L_L_m_eta_i'].append([np.mean(dict_sample['L_etai_map'][etai])])
    m_eta = m_eta + np.mean(dict_sample['L_etai_map'][etai])
m_eta = m_eta/len(dict_sample['L_L_ig_etai'])
dict_user['L_m_c'].append(np.mean(dict_sample['c_map']))
dict_user['L_m_mass'].append(m_eta+np.mean(dict_sample['c_map']))

# ------------------------------------------------------------------------------------------------------------------------------------------ #
# Performances

dict_user['move_pf'] = 0
dict_user['comp_con_vol'] = 0
dict_user['comp_kc'] = 0
dict_user['comp_as'] = 0
dict_user['comp_ed'] = 0
dict_user['comp_dt'] = 0
dict_user['write_i'] = 0
dict_user['solve_pf'] = 0
dict_user['sort_pf'] = 0
dict_user['read_pf'] = 0
dict_user['comp_vertices'] = 0
dict_user['plot_shape'] = 0
dict_user['save_dem'] = 0
dict_user['solve_dem'] = 0
dict_user['sort_dem'] = 0
dict_user['read_dem'] = 0
dict_user['plot_n_vertices'] = 0
dict_user['plot_s_m_etai_c'] = 0
dict_user['plot_perf'] = 0
dict_user['plot_d_s_a'] = 0
dict_user['plot_map'] = 0

# ------------------------------------------------------------------------------------------------------------------------------------------ #
# PFDEM iteration

dict_sample['i_DEMPF_ite'] = 0
while dict_sample['i_DEMPF_ite'] < dict_user['n_DEMPF_ite']:
    dict_sample['i_DEMPF_ite'] = dict_sample['i_DEMPF_ite'] + 1
    print('\nStep',dict_sample['i_DEMPF_ite'],'/',dict_user['n_DEMPF_ite'],'\n')

    # DEM
    run_yade(dict_user, dict_sample)

    # DEM->PF, PF, PF->DEM
    run_moose(dict_user, dict_sample)

    raise ValueError('Stop')

    # Evolution of sum and mean of etai + c
    tic_tempo = time.perf_counter() # compute performances
    plot_sum_mean_etai_c(dict_user, dict_sample) # from tools.py
    tac_tempo = time.perf_counter() # compute performances
    dict_user['plot_s_m_etai_c'] = dict_user['plot_s_m_etai_c'] + tac_tempo-tic_tempo 
    
    # plot performances
    tic_tempo = time.perf_counter() # compute performances
    plot_performances(dict_user, dict_sample) # from tools.py
    tac_tempo = time.perf_counter() # compute performances
    dict_user['plot_perf'] = dict_user['plot_perf'] + tac_tempo-tic_tempo 
    
    # plot displacement, strain, fit with Andrade law
    tic_tempo = time.perf_counter() # compute performances
    plot_disp_strain_andrade(dict_user, dict_sample) # from tools.py
    tac_tempo = time.perf_counter() # compute performances
    dict_user['plot_d_s_a'] = dict_user['plot_d_s_a'] + tac_tempo-tic_tempo 

    # plot maps configuration
    tic_tempo = time.perf_counter() # compute performances
    plot_maps_configuration(dict_user, dict_sample) # from tools.py
    tac_tempo = time.perf_counter() # compute performances
    dict_user['plot_map'] = dict_user['plot_map'] + tac_tempo-tic_tempo 

# ------------------------------------------------------------------------------------------------------------------------------------------ #
# close simulation

# save
with open('dict/dict_user', 'wb') as handle:
    pickle.dump(dict_user, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('dict/dict_sample', 'wb') as handle:
    pickle.dump(dict_sample, handle, protocol=pickle.HIGHEST_PROTOCOL)

# compute performances
tac = time.perf_counter()
hours = (tac-tic)//(60*60)
minutes = (tac-tic - hours*60*60)//(60)
seconds = int(tac-tic - hours*60*60 - minutes*60)
print("\nSimulation time : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds")
print('Simulation ends')

# sort files
reduce_n_vtk_files(dict_user, dict_sample) # from tools.py

# copy and paste to Data folder
if dict_user['save_simulation']:
    os.mkdir('../Data_PressureSolution_2G_2D/'+name)
    shutil.copytree('dict', '../Data_PressureSolution_2G_2D/'+name+'/dict')
    shutil.copytree('plot', '../Data_PressureSolution_2G_2D/'+name+'/plot')
    shutil.copy('dem_base.py', '../Data_PressureSolution_2G_2D/'+name+'/dem_base.py')
    shutil.copy('dem_to_pf.py', '../Data_PressureSolution_2G_2D/'+name+'/dem_to_pf.py')
    shutil.copy('dem_ic_base.py', '../Data_PressureSolution_2G_2D/'+name+'/dem_ic_base.py')
    shutil.copy('ic.py', '../Data_PressureSolution_2G_2D/'+name+'/ic.py')
    shutil.copy('main.py', '../Data_PressureSolution_2G_2D/'+name+'/main.py')
    shutil.copy('Parameters.py', '../Data_PressureSolution_2G_2D/'+name+'/Parameters.py')
    shutil.copy('pf_base.i', '../Data_PressureSolution_2G_2D/'+name+'/pf_base.i')
    shutil.copy('pf_to_dem.py', '../Data_PressureSolution_2G_2D/'+name+'/pf_to_dem.py')
    shutil.copy('tools.py', '../Data_PressureSolution_2G_2D/'+name+'/tools.py')


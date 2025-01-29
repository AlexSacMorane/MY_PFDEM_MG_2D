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
    # from dem to pf
    # compute mass
    compute_mass(dict_user, dict_sample) # from tools.py
    tic_dem_to_pf = time.perf_counter() # compute dem_to_pf performances
    tic_tempo = time.perf_counter() # compute performances
    move_phasefield(dict_user, dict_sample) # in dem_to_pf.py
    tac_tempo = time.perf_counter() # compute performances
    dict_user['move_pf'] = dict_user['move_pf'] + tac_tempo-tic_tempo 
    # check mass
    compute_mass_loss(dict_user, dict_sample, 'L_loss_move_pf') # from tools.py
    
    # sort eta variables into phi variables
    sort_phase_variable(dict_user, dict_sample) # in dem_to_pf.py
    # write pf file
    for i_phi in range(len(dict_sample['L_phii_map'])):
        write_array_txt(dict_sample, 'eta_'+str(i_phi+1), dict_sample['L_phii_map'][i_phi]) # from dem_to_pf.py

    # from dem to pf
    # compute mass
    compute_mass(dict_user, dict_sample) # from tools.py
    # compute diffusivity map
    tic_tempo = time.perf_counter() # compute performances
    compute_kc(dict_user, dict_sample) # in dem_to_pf.py
    tac_tempo = time.perf_counter() # compute performances
    dict_user['comp_kc'] = dict_user['comp_kc'] + tac_tempo-tic_tempo 
    # check mass
    compute_mass_loss(dict_user, dict_sample, 'L_loss_kc') # from tools.py
    
    # compute activity map
    tic_tempo = time.perf_counter() # compute performances
    compute_contact(dict_user, dict_sample) # in dem_to_pf.py
    plot_contact(dict_user, dict_sample) # in tools.py
    compute_as(dict_user, dict_sample) # in dem_to_pf.py
    tac_tempo = time.perf_counter() # compute performances
    dict_user['comp_as'] = dict_user['comp_as'] + tac_tempo-tic_tempo 
    
    # compute mass
    compute_mass(dict_user, dict_sample) # from tools.py

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
    
    # compute eta maps from the phi maps
    extract_etas_from_phis(dict_user, dict_sample) # in pf_to_dem.py
    
    # check mass
    compute_mass_loss(dict_user, dict_sample, 'L_loss_pf') # from tools.py
    
    # check integrity of etas 
    check_etas(dict_user, dict_sample) # in pf_to_dem.py

# ------------------------------------------------------------------------------------------------------------------------------------------ #

def run_yade(dict_user, dict_sample):
    '''
    Prepare and run yade simulation.
    '''
    # from pf to dem
    tic_pf_to_dem = time.perf_counter() # compute pf_to_dem performances
    compute_levelset(dict_user, dict_sample) # from pf_to_dem.py
    tac_pf_to_dem = time.perf_counter() # compute pf_to_dem performances
    dict_user['L_t_pf_to_dem_1'].append(tac_pf_to_dem-tic_pf_to_dem)
    dict_user['comp_ls'] = dict_user['comp_ls'] + tac_pf_to_dem-tic_pf_to_dem 
    
    # transmit data
    tic_tempo = time.perf_counter() # compute performances
    dict_save = {
    'E': dict_user['E'],
    'Poisson': dict_user['Poisson'],
    'kn': dict_user['kn_dem'],
    'ks': dict_user['ks_dem'],
    'n_ite_max': dict_user['n_ite_max'],
    'i_DEMPF_ite': dict_sample['i_DEMPF_ite'],
    'L_pos_w': dict_sample['L_pos_w'],
    'force_applied': dict_user['force_applied'],
    'n_steady_state_detection': dict_user['n_steady_state_detection'],
    'steady_state_detection': dict_user['steady_state_detection'],
    'print_all_dem': 'all_dem' in dict_user['L_figures'],
    'print_dem': 'dem' in dict_user['L_figures'],
    'print_vtk': 'yade_vtk' in dict_user['L_figures'],
    'extrude_z': dict_user['extrude_z'],
    'm_size': dict_user['m_size']
    }
    with open('data/main_to_dem.data', 'wb') as handle:
        pickle.dump(dict_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tac_tempo = time.perf_counter() # compute performances
    dict_user['save_dem'] = dict_user['save_dem'] + tac_tempo-tic_tempo 
    
    if 'yade_vtk' in dict_user['L_figures']:
        create_folder('vtk/yade') # from tools.py

    # dem
    print('Running DEM')
    tic_dem = time.perf_counter() # compute dem performances
    os.system('yadedaily -j '+str(dict_user['n_proc'])+' -x -n dem_base.py')
    #os.system('yadedaily -j '+str(dict_user['n_proc'])+' dem_base.py')
    tac_dem = time.perf_counter() # compute dem performances
    dict_user['L_t_dem'].append(tac_dem-tic_dem)
    dict_user['solve_dem'] = dict_user['solve_dem'] + tac_dem-tic_dem 

    # sort files
    #sort_dem_files(dict_user, dict_sample) # from dem_to_pf.py

    # load data
    tic_tempo = time.perf_counter() # compute performances
    with open('data/dem_to_main.data', 'rb') as handle:
        dict_save = pickle.load(handle)
    L_contact = dict_save['L_contact']

    # iterate on potential contact
    ij = 0
    for i_grain in range(len(dict_sample['L_etai_map'])-1):
        for j_grain in range(i_grain+1, len(dict_sample['L_etai_map'])):
            if len(L_contact) > 0:
                i_contact = 0
                contact_found = L_contact[i_contact][0:2] == [i_grain, j_grain]
                while not contact_found and i_contact < len(L_contact)-1:
                    i_contact = i_contact + 1
                    contact_found = L_contact[i_contact][0:2] == [i_grain, j_grain]
            else :
                contact_found = False
            if dict_sample['i_DEMPF_ite'] == 1:
                if contact_found:
                    dict_user['L_L_overlap'].append([L_contact[i_contact][2]])
                    dict_user['L_L_normal_force'].append([L_contact[i_contact][3]])
                else:
                    dict_user['L_L_overlap'].append([0])
                    dict_user['L_L_normal_force'].append([np.array([0,0,0])])
            else :
                if contact_found:
                    dict_user['L_L_overlap'][ij].append(L_contact[i_contact][2])
                    dict_user['L_L_normal_force'][ij].append(L_contact[i_contact][3])
                else:
                    dict_user['L_L_overlap'][ij].append(0)
                    dict_user['L_L_normal_force'][ij].append(np.array([0,0,0]))
            ij = ij + 1
    plot_dem(dict_user, dict_sample) # from tools.py
    tac_tempo = time.perf_counter() # compute performances
    dict_user['read_dem'] = dict_user['read_dem'] + tac_tempo-tic_tempo 
        
# ------------------------------------------------------------------------------------------------------------------------------------------ #
# Plan
    
# get parameters
dict_user = get_parameters() # from Parameters.py
dict_sample = {}

# folders
create_folder('vtk') # from tools.py
create_folder('plot') # from tools.py
if 'configuration_eta' in dict_user['L_figures'] or\
   'configuration_c' in dict_user['L_figures'] or\
   'configuration_ic' in dict_user['L_figures']:
    create_folder('plot/configuration') # from tools.py
if 'all_dem' in dict_user['L_figures']:
    create_folder('plot/dem') # from tools.py
if 'before_after' in dict_user['L_figures']:
    create_folder('plot/before_after') # from tools.py
create_folder('data') # from tools.py
create_folder('input') # from tools.py
create_folder('dict') # from tools.py

# compute performances
tic = time.perf_counter()

# ------------------------------------------------------------------------------------------------------------------------------------------ #
# Create initial condition

# generate grains and solute
load_microstructure(dict_user, dict_sample) # from ic.py
create_solute(dict_user, dict_sample) # from ic.py

# save initial shape
save_initial_shape(dict_user, dict_sample) # from tools.py

# plot
plot_config_ic(dict_user, dict_sample) # from tools.py

# compute tracker
plot_sum_mean_etai_c(dict_user, dict_sample) # from tools.py

# check if the mesh map is inside the database
check_mesh_database(dict_user, dict_sample) # from tools.py

# ------------------------------------------------------------------------------------------------------------------------------------------ #
# Performances

dict_user['move_pf'] = 0
dict_user['comp_con_vol'] = 0
dict_user['comp_kc'] = 0
dict_user['comp_as'] = 0
dict_user['write_i'] = 0
dict_user['solve_pf'] = 0
dict_user['sort_pf'] = 0
dict_user['read_pf'] = 0
dict_user['comp_ls'] = 0
dict_user['save_dem'] = 0
dict_user['solve_dem'] = 0
dict_user['read_dem'] = 0
dict_user['plot_s_m_etai_c'] = 0
dict_user['plot_perf'] = 0

# ------------------------------------------------------------------------------------------------------------------------------------------ #
# PFDEM iteration


dict_sample['i_DEMPF_ite'] = 0
while dict_sample['i_DEMPF_ite'] < dict_user['n_DEMPF_ite']:
    dict_sample['i_DEMPF_ite'] = dict_sample['i_DEMPF_ite'] + 1
    print('\nStep',dict_sample['i_DEMPF_ite'],'/',dict_user['n_DEMPF_ite'],'\n')

    # print configuration c and eta_i
    plot_config(dict_user, dict_sample) # from tools

    # DEM
    run_yade(dict_user, dict_sample)

    # DEM->PF, PF, PF->DEM
    run_moose(dict_user, dict_sample)

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

# ------------------------------------------------------------------------------------------------------------------------------------------ #
# close simulation

# save final shape
save_final_shape(dict_user, dict_sample) # from tools.py

# clean dict
del dict_sample['L_phi_L_etas']
del dict_sample['L_phi_L_boxs']
del dict_sample['L_phii_map']
del dict_sample['kc_map']
del dict_sample['L_vol_contact']
del dict_sample['L_surf_contact']
del dict_sample['as_map']

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

# save mesh database 
save_mesh_database(dict_user, dict_sample) # from tools.py

# sort files
#reduce_n_vtk_files(dict_user, dict_sample) # from tools.py
# functions needed to be adapted

# delete folder
shutil.rmtree('data')
shutil.rmtree('input')

#output
print('\n')
print('mass csv:', dict_user['L_sum_mass'][0], '->', dict_user['L_sum_mass'][-1])
print('=', 100*abs(dict_user['L_sum_mass'][0]-dict_user['L_sum_mass'][-1])/dict_user['L_sum_mass'][0], '%')


# -*- encoding=utf-8 -*-

from pathlib import Path
import numpy as np
import math, skfmm, pickle, os

# ------------------------------------------------------------------------------------------------------------------------------------------ #

def load_microstructure(dict_user, dict_sample):
    '''
    Load microstructure as initial conditions.
    Mesh and phase field maps are generated
    '''    
    # ------------------------------------------------------------------------------------------------------------------------------------------ #
    # Load data and apply initial force

    if not Path('ms/from_ic.data').exists():
        # transmit data from main
        dict_save = {
            'Poisson': dict_user['Poisson'],
            'E': dict_user['E'],
            'kn': dict_user['kn_dem'],
            'ks': dict_user['ks_dem'],
            'force_applied': dict_user['force_applied']
            }
        with open('ms/from_main_to_ic.data', 'wb') as handle:
            pickle.dump(dict_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # run simulation 
        os.system('yadedaily -j '+str(dict_user['n_proc'])+' -x -n dem_base_ic.py')
        #os.system('yadedaily -j '+str(dict_user['n_proc'])+' dem_base_ic.py')

        print('end of the initialization\n')

    # ------------------------------------------------------------------------------------------------------------------------------------------ #
    # Load data
    with open('ms/from_ic.data', 'rb') as handle:
        dict_save = pickle.load(handle)

    # update dict
    dict_user['extrude_z'] = dict_save['extrude_z']
    dict_user['margins'] = dict_save['margins']
    dict_sample['L_pos_w'] = []
    for pos_w in dict_save['L_pos_w']:
        dict_sample['L_pos_w'].append(pos_w*1e-6/dict_user['n_dist'])

    # ------------------------------------------------------------------------------------------------------------------------------------------ #
    # Compute mesh dependent data

    # mesh size
    dict_user['m_size'] = dict_save['m_size']*1e-6/dict_user['n_dist']
    # phase-field interface width
    dict_user['w_int'] = dict_user['m_size']*dict_user['n_int']
    # phase-field gradient coefficient
    dict_user['kappa_eta'] = dict_user['Energy_barrier']*dict_user['w_int']*dict_user['w_int']/9.86
    # phase-field kinetics of dissolution/precipitation
    dict_user['k_diss'] = dict_user['k_diss']*(0.005)/(dict_user['m_size']) # ed_j = ed_i*m_i/m_j 
    dict_user['k_prec'] = dict_user['k_prec']*(0.005)/(dict_user['m_size']) # ed_j = ed_i*m_i/m_j 
    # size fluid film
    dict_user['size_film'] = dict_user['m_size']*dict_user['n_size_film']
    # solute diffusion coefficient
    dict_user['D_solute'] = dict_user['D_solute']/2/dict_user['size_film'] 

    # ------------------------------------------------------------------------------------------------------------------------------------------ #
    # Create initial mesh
    print("Creating initial mesh")

    # limits
    x_L_min = dict_sample['L_pos_w'][0] -2*dict_user['w_int']
    x_L_max = dict_sample['L_pos_w'][1] +2*dict_user['w_int']
    y_L_min = dict_sample['L_pos_w'][2] -2*dict_user['w_int']
    y_L_max = dict_sample['L_pos_w'][3] +2*dict_user['w_int']
    
    # mesh
    x_L = np.arange(x_L_min, x_L_max+dict_user['m_size']*0.1, dict_user['m_size'])
    y_L = np.arange(y_L_min, y_L_max+dict_user['m_size']*0.1, dict_user['m_size'])

    # ------------------------------------------------------------------------------------------------------------------------------------------ #
    # Create Phase-Field 
    
    print("Creating initial phase field maps")
    L_etai_map = []

    # iterate on grain
    for i_data in range(dict_save['n_grains']):
        
        # load data
        with open('ms/from_ic_grain_'+str(i_data)+'.data', 'rb') as handle:
            dict_save = pickle.load(handle)
        x_L_i = []
        for x in dict_save['x_L']:
            x_L_i.append(x*1e-6/dict_user['n_dist'])
        y_L_i = []
        for y in dict_save['y_L']:
            y_L_i.append(y*1e-6/dict_user['n_dist'])
        ls_map_i = dict_save['ls_map']
        
        # Create binary map  
        bin_i_map = -np.ones((len(y_L), len(x_L)))
        
        # iteration on x
        for j_x in range(len(x_L_i)):
            # iteration on y
            for j_y in range(len(y_L_i)):
                
                # find j (local) in i (global): nearest node
                L_search = list(abs(np.array(x_L-x_L_i[j_x])))
                i_x = L_search.index(min(L_search))
                L_search = list(abs(np.array(y_L-y_L_i[j_y])))
                i_y = L_search.index(min(L_search))
          
                # compute the binary map
                if ls_map_i[-1-j_y, j_x] < 0:
                    bin_i_map[-1-i_y, i_x] = 1
                else:
                    bin_i_map[-1-i_y, i_x] = -1

        # compute the signed distance function
        sdf_i_map = -skfmm.distance(bin_i_map, dx=np.array([dict_user['m_size'], dict_user['m_size']]))

        # compute phase variable
        eta_i_map = np.zeros((len(y_L), len(x_L)))
        # iteration on x
        for i_x in range(len(x_L)):
            # iteration on y
            for i_y in range(len(y_L)):
                sdf = sdf_i_map[-1-i_y, i_x]
                # cosine profile
                if sdf > dict_user['w_int']/2 :
                    eta_i_map[-1-i_y, i_x] = 0
                elif sdf < -dict_user['w_int']/2 :
                    eta_i_map[-1-i_y, i_x] = 1
                else :
                    eta_i_map[-1-i_y, i_x] = 0.5*(1+math.cos(math.pi*(sdf+dict_user['w_int']/2)/dict_user['w_int']))
        
        # save
        L_etai_map.append(eta_i_map)

    # save dict
    dict_sample['L_etai_map'] = L_etai_map
    dict_sample['x_L'] = x_L
    dict_sample['y_L'] = y_L
    dict_user['n_mesh_x'] = len(x_L)
    dict_user['n_mesh_y'] = len(y_L)
 
# ------------------------------------------------------------------------------------------------------------------------------------------ #

def create_solute(dict_user, dict_sample):
    '''
    Create the map of the solute distribution.
    '''
    c_map = np.zeros((dict_user['n_mesh_y'], dict_user['n_mesh_x']))
    for i_x in range(len(dict_sample['x_L'])):
        for i_y in range(len(dict_sample['y_L'])):
            c_map[-1-i_y, i_x] = dict_user['C_eq'] # system at the equilibrium initialy
    # save in dict
    dict_sample['c_map'] = c_map

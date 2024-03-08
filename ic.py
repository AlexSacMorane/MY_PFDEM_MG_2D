# -*- encoding=utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math, skfmm, random, time, pickle, os

# ------------------------------------------------------------------------------------------------------------------------------------------ #

def create_spheres(dict_user, dict_sample):
    '''
    Create initial conditions with multiple spheres.
    Mesh and phase field maps are generated
    '''
    # Generate polyhedra particules
    if dict_user['Shape'] == 'Sphere_no_overlap':
        L_center, L_radius = generate_sphere_no_overlap_tempo(dict_user, dict_sample)
    elif dict_user['Shape'] == 'Sphere_cfc':
        L_center, L_radius = generate_sphere_cfc_tempo(dict_user, dict_sample)
    # save
    dict_sample['L_radius'] = L_radius

    # transmit data
    dict_save = {
    'E': dict_user['E'],
    'Poisson': dict_user['Poisson'],
    'force_applied': dict_user['force_applied'],
    'k_control_force': dict_user['k_control_force'],
    'd_y_limit': dict_user['d_y_limit'],
    'L_center': L_center,
    'L_radius': L_radius,
    'n_phi': dict_user['n_phi'],
    'x_min_wall': dict_user['x_min_wall'],
    'x_max_wall': dict_user['x_max_wall'],
    'y_min_wall': dict_user['y_min_wall'],
    'y_max_wall': dict_user['y_max_wall'],
    'n_ite_max': dict_user['n_ite_max'],
    'steady_state_detection': dict_user['steady_state_detection'],
    'n_steady_state_detection': dict_user['n_steady_state_detection'],
    'print_dem_ic': 'ic_dem' in dict_user['L_figures']
    }
    with open('data/main_to_dem_ic.data', 'wb') as handle:
        pickle.dump(dict_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Compute ic configuration with yade
    os.system('yadedaily -j '+str(dict_user['n_proc'])+' -x -n dem_ic_base.py')
    #os.system('yadedaily -j '+str(dict_user['n_proc'])+' dem_ic_base.py')
    # Load data
    with open('data/dem_ic_to_main.data', 'rb') as handle:
        dict_save = pickle.load(handle)
    # save
    dict_sample['L_center'] = dict_save['L_pos']
    dict_sample['L_box'] = dict_save['L_box']
    # tracker
    dict_user['L_sample_height'].append(dict_save['y_max_wall']-dict_user['y_min_wall'])

    # Plot ic configuration
    if 'ic_config' in dict_user['L_figures']:
        fig, ((ax1)) = plt.subplots(nrows=1,ncols=1,figsize=(16,9))
        for i_g in range(len(L_radius)):
            L_x = []
            L_y = []
            for i_phi in range(dict_user['n_phi']):
                phi = 2*math.pi*i_phi/dict_user['n_phi']
                L_x.append(dict_sample['L_center'][i_g][0]+math.cos(phi)*L_radius[i_g])
                L_y.append(dict_sample['L_center'][i_g][1]+math.sin(phi)*L_radius[i_g])
            L_x.append(L_x[0])
            L_y.append(L_y[0])
            ax1.plot(L_x, L_y)
        ax1.plot([dict_user['x_min_wall'], dict_user['x_max_wall'], dict_user['x_max_wall'], dict_user['x_min_wall'], dict_user['x_min_wall']],\
                 [dict_user['y_min_wall'], dict_user['y_min_wall'], dict_save['y_max_wall'], dict_save['y_max_wall'], dict_user['y_min_wall']],\
                 color='k')
        ax1.set_title(r'Initial Configuration')
        ax1.axis('equal')
        fig.tight_layout()
        fig.savefig('plot/ic_config.png')
        plt.close(fig)

    # Generate mesh
    print("\nInitial mesh generation")
    x_min = dict_user['x_min_wall'] - dict_user['margin_domain']
    x_max = dict_user['x_max_wall'] + dict_user['margin_domain']
    y_min = dict_user['y_min_wall'] - dict_user['margin_domain']
    y_max = dict_user['y_max_wall'] + dict_user['margin_domain']
    x_L = np.arange(x_min, x_max, dict_user['size_mesh_x'])
    y_L = np.arange(y_min, y_max, dict_user['size_mesh_y'])
    dict_sample['x_L'] = x_L
    dict_sample['y_L'] = y_L

    # Distribute pf
    distribute_pf_variable(dict_user, dict_sample)
    # Plot ic configursation
    if 'ic_pf_dist' in dict_user['L_figures']:
        L_color = ['r','b','g','m','c','y']
        fig, ((ax1)) = plt.subplots(nrows=1,ncols=1,figsize=(16,9))
        for i_g in range(len(L_radius)):
            L_x = []
            L_y = []
            for i_phi in range(dict_user['n_phi']):
                phi = 2*math.pi*i_phi/dict_user['n_phi']
                L_x.append(dict_sample['L_center'][i_g][0]+math.cos(phi)*L_radius[i_g])
                L_y.append(dict_sample['L_center'][i_g][1]+math.sin(phi)*L_radius[i_g])
            L_x.append(L_x[0])
            L_y.append(L_y[0])
            etai = 0
            while i_g not in dict_sample['L_L_ig_etai'][etai]:
                etai = etai + 1
            ax1.plot(L_x, L_y, color=L_color[etai])
        ax1.plot([dict_user['x_min_wall'], dict_user['x_max_wall'], dict_user['x_max_wall'], dict_user['x_min_wall'], dict_user['x_min_wall']],\
                 [dict_user['y_min_wall'], dict_user['y_min_wall'], dict_save['y_max_wall'], dict_save['y_max_wall'], dict_user['y_min_wall']],\
                 color='k')
        ax1.set_title(r'PF distribution')
        ax1.axis('equal')
        fig.tight_layout()
        fig.savefig('plot/ic_pf_distribution.png')
        plt.close(fig)
    
    # Generate individual pf
    generate_pf_variable(dict_user, dict_sample)

# ------------------------------------------------------------------------------------------------------------------------------------------ #

def generate_sphere_no_overlap_tempo(dict_user, dict_sample):
    '''
    Generate spheres (polyhedra) in the box.

    Not overlapping condition is applied.
    '''
    # initialization
    L_radius = []
    L_center = []
    # iteration on the number of grain
    for i_g in range(dict_user['n_grain']):
        grain_generated = False
        i_gen = 0
        while i_gen < dict_user['n_max_gen_ic'] and not grain_generated:
            i_gen = i_gen + 1
            # random generation of the grain
            radius_tempo = dict_user['radius']*(1+dict_user['var_radius']*(random.random()-0.5)*2)
            center_x_tempo = random.random()*(dict_user['x_max_wall']-radius_tempo-(dict_user['x_min_wall']+radius_tempo))+dict_user['x_min_wall']+radius_tempo
            center_y_tempo = random.random()*(dict_user['y_max_wall']-radius_tempo-(dict_user['y_min_wall']+radius_tempo))+dict_user['y_min_wall']+radius_tempo
            # check if the tempo grain can be added
            if L_center == []: # empty list
                L_radius.append(radius_tempo) # add grain
                L_center.append(np.array([center_x_tempo, center_y_tempo]))
                grain_generated = True
            else: # not empty list 
                # check not overlaping condition
                overlap = False
                for j_g in range(len(L_radius)):
                    distance = np.linalg.norm(L_center[j_g]-np.array([center_x_tempo, center_y_tempo]))
                    if distance < radius_tempo + L_radius[j_g]:
                        overlap = True
                if not overlap:                    
                    L_radius.append(radius_tempo) # add grain
                    L_center.append(np.array([center_x_tempo, center_y_tempo]))
                    grain_generated = True

    # user GUI 
    print('\nIC generation')
    print(f"{len(L_radius)} grains generated / {dict_user['n_grain']}\n")

    return L_center, L_radius

# ------------------------------------------------------------------------------------------------------------------------------------------ #

def generate_sphere_cfc_tempo(dict_user, dict_sample):
    '''
    Generate spheres (polyhedra) in the box.

    A cfc mesh is assumed.
    '''
    # initialization
    L_radius = []
    L_center = []
    # iteration on the line
    for i_y in range(dict_user['n_lines']):
        if i_y%2 == 0:
            adaptation = 0 
        else : 
            adaptation = 1
        # iteration on the grains of the line
        for i_x in range(dict_user['n_grains_x']-adaptation):
            # random generation of the grain radius
            radius_tempo = dict_user['radius']*(1+dict_user['var_radius']*(random.random()-0.5)*2)
            # random generation of the grain center (based on cfc mesh)
            if i_y%2 == 0:
                center_base_x = i_x*2*dict_user['radius'] + dict_user['radius'] 
            else : 
                center_base_x = i_x*2*dict_user['radius'] + 2*dict_user['radius']
            center_base_y = i_y*2*dict_user['radius'] + dict_user['radius']
            center_x = center_base_x + dict_user['radius']*dict_user['var_pos']*(random.random()-0.5)*2
            center_y = center_base_y + dict_user['radius']*dict_user['var_pos']*(random.random()-0.5)*2
            L_radius.append(radius_tempo) # add grain
            L_center.append(np.array([center_x, center_y]))

    # user GUI 
    print('\nIC generation')
    print(f"{len(L_radius)} grains generated\n")

    return L_center, L_radius

# ------------------------------------------------------------------------------------------------------------------------------------------ #

def distribute_pf_variable(dict_user, dict_sample):
    '''
    Assign grains to etai (pf variables).

    A minimal distance is set based on a security factor.
    '''
    print('\nPF distribution')
    # initialization
    L_ig_near = []
    L_eta_near = []
    for i_g in range(len(dict_sample['L_radius'])):
        L_ig_near.append([])
        L_eta_near.append([])
    # iterate on grains to detecte proximity
    for i_g1 in range(1, len(dict_sample['L_radius'])):
        for i_g2 in range(0, i_g1):
            if np.linalg.norm(dict_sample['L_center'][i_g1]-dict_sample['L_center'][i_g2]) < \
               dict_user['factor_pf']*(dict_sample['L_radius'][i_g1]+dict_sample['L_radius'][i_g2]):
                L_ig_near[i_g1].append(i_g2)
                L_ig_near[i_g2].append(i_g1)
    # etai distribution - first try
    # initialization with the first grain
    L_n_etai = [1]
    L_etai = [0]
    L_L_ig_etai = [[0]]
    L_etai_ig = [0]
    for i_g in range(len(dict_sample['L_radius'])):
        if 0 in L_ig_near[i_g]:
            L_eta_near[i_g].append(0)
    # iteration on the others grainss
    for i_g in range(1, len(dict_sample['L_radius'])):
        etai_defined = False
        etai = 0
        # try to use a already created pf variable
        while etai < len(L_etai) and not etai_defined:
            if etai not in L_eta_near[i_g]:
                L_L_ig_etai[etai].append(i_g)
                L_n_etai[etai] = L_n_etai[etai] + 1 
                L_etai_ig.append(etai)
                etai_defined = True
                for j_g in range(len(dict_sample['L_radius'])):
                    if i_g in L_ig_near[j_g]:
                        L_eta_near[j_g].append(etai)
            etai = etai + 1
        # creation of a new pf variable
        if etai == len(L_etai) and not etai_defined:
            L_etai.append(etai)
            L_L_ig_etai.append([i_g])
            L_n_etai.append(1)
            L_etai_ig.append(etai)
            for j_g in range(len(dict_sample['L_radius'])):
                if i_g in L_ig_near[j_g]:
                    L_eta_near[j_g].append(etai)
    # etai distribution - adaptation
    # try to have the same number of grain assigned to all eta
    m_n_etai = np.mean(L_n_etai)
    adaptation_done = False
    adaptation_i = 0
    while not adaptation_done :
        adaptation_i = adaptation_i + 1
        etai_over = L_n_etai.index(max(L_n_etai))
        etai_under = L_n_etai.index(min(L_n_etai))
        L_ig_over = L_L_ig_etai[etai_over]
        ig_to_work = L_ig_over[random.randint(0,len(L_ig_over)-1)]
        # check if etai_under is available for grain ig
        if etai_under not in L_eta_near[ig_to_work]:
            L_n_etai[etai_over] = L_n_etai[etai_over] - 1
            L_n_etai[etai_under] = L_n_etai[etai_under] + 1
            L_L_ig_etai[etai_over].remove(ig_to_work)
            L_L_ig_etai[etai_under].append(ig_to_work)
            L_etai_ig[ig_to_work] = etai_under
            for j_g in range(len(dict_sample['L_radius'])):
                if ig_to_work in L_ig_near[j_g]:
                    L_eta_near[j_g].remove(etai_over)
                    L_eta_near[j_g].append(etai_under)
        # check the quality of the distribution
        adaptation_done = True
        # window of 1 around the mean value 
        for n_etai in L_n_etai :
            if m_n_etai-2 < n_etai and n_etai < m_n_etai+2:
                adaptation_done = False
        # limit the number of tries
        if adaptation_i > len(dict_sample['L_radius']):
            adaptation_done = True
    # save
    dict_sample['L_L_ig_etai'] = L_L_ig_etai
    dict_sample['L_etai_ig'] = L_etai_ig
    # user GUI 
    print(f"{len(L_etai)} phase variables used.")
    print(f"{round(len(dict_sample['L_radius'])/len(L_etai),1)} grains described for one phase variable.\n")
        
# ------------------------------------------------------------------------------------------------------------------------------------------ #

def generate_pf_variable(dict_user, dict_sample):
    '''
    Compute etai (pf variables).
    '''
    # initialization
    L_etai_map = []
    # iterate on pf variables
    for etai in range(len(dict_sample['L_L_ig_etai'])):
        L_etai_map.append(np.zeros((len(dict_sample['y_L']), len(dict_sample['x_L']))))
    
    # iteration on x
    for i_x in range(len(dict_sample['x_L'])):
        x = dict_sample['x_L'][i_x]
        # iteration on y
        for i_y in range(len(dict_sample['y_L'])):
            y = dict_sample['y_L'][i_y]
            # iteration on pf variables
            for etai in range(len(dict_sample['L_L_ig_etai'])):
                # look for the nearest center
                min_distance = None
                # iterate on grain of this pf variable
                for ig in dict_sample['L_L_ig_etai'][etai]:
                    # distance to gi
                    d_node_to_gi = np.linalg.norm(np.array([x,y])-np.array(dict_sample['L_center'][ig]))
                    if min_distance == None:
                        min_distance = d_node_to_gi
                        ig_nearest = ig
                    else : 
                        if d_node_to_gi < min_distance:
                            min_distance = d_node_to_gi
                            ig_nearest = ig
                # compute etai
                if min_distance <= dict_sample['L_radius'][ig_nearest]-dict_user['w_int']/2 :
                    L_etai_map[etai][-1-i_y, i_x] = 1
                elif dict_sample['L_radius'][ig_nearest]-dict_user['w_int']/2 < min_distance and\
                     min_distance < dict_sample['L_radius'][ig_nearest]+dict_user['w_int']/2:
                    L_etai_map[etai][-1-i_y, i_x] = 0.5*(1+math.cos(math.pi*(min_distance-dict_sample['L_radius'][ig_nearest]+dict_user['w_int']/2)/dict_user['w_int']))
                elif dict_sample['L_radius'][ig_nearest]+dict_user['w_int']/2 <= min_distance :
                    L_etai_map[etai][-1-i_y, i_x] = 0
    # save
    dict_sample['L_etai_map'] = L_etai_map
    # plot
    if 'ic_pf_map' in dict_user['L_figures']:
        for etai in range(len(dict_sample['L_L_ig_etai'])):
            fig, (ax1) = plt.subplots(1,1,figsize=(16,9))    
            im = ax1.imshow(L_etai_map[etai], interpolation = 'nearest', extent=(dict_sample['x_L'][0],dict_sample['x_L'][-1],dict_sample['y_L'][0],dict_sample['y_L'][-1]))
            fig.colorbar(im, ax=ax1)
            ax1.set_title('Initial map of eta '+str(etai),fontsize = 30)
            fig.tight_layout()
            fig.savefig('plot/ic_map_eta_'+str(etai)+'.png')
            plt.close(fig)
    # user GUI 
    print("Initial phase maps generation\n")

#------------------------------------------------------------------------------------------------------------------------------------------ #

def create_solute(dict_user, dict_sample):
    '''
    Create the map of the solute distribution.
    '''
    c_map = np.zeros((len(dict_sample['y_L']), len(dict_sample['x_L'])))
    for i_x in range(len(dict_sample['x_L'])):
        for i_y in range(len(dict_sample['y_L'])):
            c_map[-1-i_y, i_x] = dict_user['C_eq'] # system at the equilibrium initialy
    # save
    dict_sample['c_map'] = c_map
    # plot
    if 'ic_c_map' in dict_user['L_figures']:
        fig, (ax1) = plt.subplots(1,1,figsize=(16,9))    
        im = ax1.imshow(c_map, interpolation = 'nearest', extent=(dict_sample['x_L'][0],dict_sample['x_L'][-1],dict_sample['y_L'][0],dict_sample['y_L'][-1]))
        fig.colorbar(im, ax=ax1)
        ax1.set_title('Initial map of the solute', fontsize = 30)
        fig.tight_layout()
        fig.savefig('plot/ic_map_solute.png')
        plt.close(fig)
    # user GUI 
    print("Initial solute map generation\n")

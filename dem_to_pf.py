# -*- encoding=utf-8 -*-

import pickle, math, os, shutil
from pathlib import Path
from scipy.ndimage import binary_dilation, label
import numpy as np
import matplotlib.pyplot as plt

# own
from tools import *
from pf_to_dem import *

# -----------------------------------------------------------------------------#

def move_phasefield(dict_user, dict_sample):
    '''
    Move phase field maps by interpolation.
    '''
    # g2 (as g1 is fixed)

    # load data
    with open('data/dem_to_main.data', 'rb') as handle:
        dict_save = pickle.load(handle)
    displacement_g2 = dict_save['displacement_g2']

    # tracker
    dict_user['L_displacement'].append(displacement_g2[1])

    # loading old variables
    eta_1_map = dict_sample['eta_1_map']
    eta_2_map = dict_sample['eta_2_map']

    # updating phase map
    print('Updating phase field maps')
    eta_1_map_new = np.zeros((dict_user['n_mesh_y'], dict_user['n_mesh_x']))
    eta_2_map_new = np.zeros((dict_user['n_mesh_y'], dict_user['n_mesh_x']))
    
    # iteration on y
    i_y_old = 0
    for i_y in range(len(dict_sample['y_L'])):
        y = dict_sample['y_L'][i_y]

        # eta 1, fixed
        eta_1_map_new = eta_1_map

        # eta 2
        if displacement_g2[1] < 0:
            if y-displacement_g2[1] <= dict_sample['y_L'][-1]:
                # look for window
                while not (dict_sample['y_L'][i_y_old] <= y-displacement_g2[1] and y-displacement_g2[1] < dict_sample['y_L'][i_y_old+1]):
                    i_y_old = i_y_old + 1
                # interpolate
                eta_2_map_new[-1-i_y, :] = (eta_2_map[-1-(i_y_old+1), :] - eta_2_map[-1-i_y_old, :])/(dict_sample['y_L'][i_y_old+1] - dict_sample['y_L'][i_y_old])*\
                                           (y-displacement_g2[1] - dict_sample['y_L'][i_y_old]) + eta_2_map[-1-i_y_old, :]
        elif displacement_g2[1] > 0:
            if dict_sample['y_L'][0] <= y-displacement_g2[1]:
                # look for window
                while not (dict_sample['y_L'][i_y_old] <= y-displacement_g2[1] and y-displacement_g2[1] < dict_sample['y_L'][i_y_old+1]):
                    i_y_old = i_y_old + 1
                # interpolate
                eta_2_map_new[-1-i_y, :] = (eta_2_map[-1-(i_y_old+1), :] - eta_2_map[-1-i_y_old, :])/(dict_sample['y_L'][i_y_old+1] - dict_sample['y_L'][i_y_old])*\
                                           (y-displacement_g2[1] - dict_sample['y_L'][i_y_old]) + eta_2_map[-1-i_y_old, :]
        else :
            eta_2_map_new = eta_2_map

    # The solute map is updated
    # the solute is push out/in of the grain
    # this is done in compute_kc() from dem_to_pf.py called later

    # update variables
    dict_sample['eta_1_map'] = eta_1_map_new
    dict_sample['eta_2_map'] = eta_2_map_new

    # write txts for phase field
    write_eta_txt(dict_user, dict_sample) # phase field

# -----------------------------------------------------------------------------#

def compute_as(dict_user, dict_sample):
    '''
    Compute activity of solid.
    '''
    # load data from dem
    with open('data/dem_to_main.data', 'rb') as handle:
        dict_save = pickle.load(handle)
    contactPoint = dict_save['contact_point']
    normalForce = dict_user['force_applied_target']
    contact_area = 1*(dict_sample['box_contact_x_max'] - dict_sample['box_contact_x_min'])

    # tracker
    dict_user['L_P_applied'].append(np.linalg.norm(normalForce)/contact_area)

    # plot
    if 'contact_pressure' in dict_user['L_figures'] :
        fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
        # overlap
        ax1.plot(dict_user['L_P_applied'])
        ax1.set_title('Pressure at the contact (Pa)',fontsize=20)
        fig.tight_layout()
        fig.savefig('plot/contact_pressure.png')
        plt.close(fig)

    # init
    dict_sample['as_map'] = np.zeros((dict_user['n_mesh_y'], dict_user['n_mesh_x']))
    # cst
    R_gas = 82.06e5 # cm3 Pa K-1 mol-1
    Temp = 25+278   # K
    V_m = 27.1      # cm3 mol-1
    conv = 1        # = 1/C_ref, normalize the sink/source
    s_mesh = 1*(dict_sample['x_L'][1]-dict_sample['x_L'][0]) # surface of a mesh, m2
    v_mesh = 1*(dict_sample['x_L'][1]-dict_sample['x_L'][0])*(dict_sample['y_L'][1]-dict_sample['y_L'][0]) # volume of a mesh, m3

    # iterate on mesh
    for i_x in range(len(dict_sample['x_L'])):
        for i_y in range(len(dict_sample['y_L'])):
            # determine pressure
            if dict_sample['eta_1_map'][i_y, i_x] > dict_user['eta_contact_box_detection'] and dict_sample['eta_2_map'][i_y, i_x] > dict_user['eta_contact_box_detection']: # in the contact
                P =  np.linalg.norm(normalForce)/contact_area # Pa
            else : # not in the contact
                P = 0 # Pa
            # save in the map
            dict_sample['as_map'][i_y, i_x] = math.exp(P*V_m/(R_gas*Temp))

    # write as
    write_as_txt(dict_user, dict_sample)

# -----------------------------------------------------------------------------#

def compute_ed(dict_user, dict_sample):
    '''
    Compute the average ed in the sample, in the contact zone, in the large contact zone and track ed value at the center of the contact.
    '''
    # constants
    s_mesh = 1*(dict_sample['x_L'][1]-dict_sample['x_L'][0])
    c_eq = 1
    k_diss = dict_user['k_diss']
    k_prec = dict_user['k_prec']
    v_mesh = 1*(dict_sample['x_L'][1]-dict_sample['x_L'][0])*(dict_sample['y_L'][1]-dict_sample['y_L'][0])
    conv = 1
    # compute the map of ed
    ed_map = np.zeros((len(dict_sample['y_L']), len(dict_sample['x_L'])))
    # compute mean ed in contact
    m_ed_contact = 0
    n_contact = 0
    m_ed_plus_contact = 0
    n_plus_contact = 0
    m_ed_minus_contact = 0
    n_minus_contact = 0
    m_ed_large_contact = 0
    n_large_contact = 0
    m_ed_plus_large_contact = 0
    n_plus_large_contact = 0
    m_ed_minus_large_contact = 0
    n_minus_large_contact = 0
    # iterate on mesh
    for i_x in range(len(dict_sample['x_L'])):
        for i_y in range(len(dict_sample['y_L'])):
            # read variables
            as_value = dict_sample['as_map'][i_y, i_x]
            c = dict_sample['c_map'][i_y, i_x]
            # compute and build map
            if c < c_eq*as_value:
                ed_value = k_diss*s_mesh*as_value*(1-c/(c_eq*as_value))/v_mesh*conv*2/3
            else :
                ed_value = k_prec*s_mesh*as_value*(1-c/(c_eq*as_value))/v_mesh*conv*2/3
            ed_map[i_y, i_x] = ed_value
            # compute mean ed in contact
            if dict_sample['eta_1_map'][i_y, i_x] > dict_user['eta_contact_box_detection'] and dict_sample['eta_2_map'][i_y, i_x] > dict_user['eta_contact_box_detection']:
                m_ed_contact = m_ed_contact + ed_value
                n_contact = n_contact + 1
                if ed_value > 0 :
                    m_ed_plus_contact = m_ed_plus_contact + ed_value
                    n_plus_contact = n_plus_contact + 1    
                else :
                    m_ed_minus_contact = m_ed_minus_contact + ed_value
                    n_minus_contact = n_minus_contact + 1   
            # compute mean ed in large contact
            if dict_sample['eta_1_map'][i_y, i_x] > 0.05 and dict_sample['eta_2_map'][i_y, i_x] > 0.05:
                m_ed_large_contact = m_ed_large_contact + ed_value
                n_large_contact = n_large_contact + 1
                if ed_value > 0 :
                    m_ed_plus_large_contact = m_ed_plus_large_contact + ed_value
                    n_plus_large_contact = n_plus_large_contact + 1    
                else :
                    m_ed_minus_large_contact = m_ed_minus_large_contact + ed_value
                    n_minus_large_contact = n_minus_large_contact + 1 
    # tracker
    dict_user['L_m_ed'].append(np.mean(ed_map))
    add_element_list(dict_user['L_m_ed_contact'], m_ed_contact, n_contact)
    add_element_list(dict_user['L_m_ed_large_contact'], m_ed_large_contact, n_large_contact)
    add_element_list(dict_user['L_m_ed_plus_contact'], m_ed_plus_contact, n_plus_contact)
    add_element_list(dict_user['L_m_ed_minus_contact'], m_ed_minus_contact, n_minus_contact)
    add_element_list(dict_user['L_m_ed_plus_large_contact'], m_ed_plus_large_contact, n_plus_large_contact)
    add_element_list(dict_user['L_m_ed_minus_large_contact'], m_ed_minus_large_contact, n_minus_large_contact)
    
    # plot
    if 'm_ed' in dict_user['L_figures']:
        fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
        ax1.plot(dict_user['L_m_ed'])
        ax1.set_title('Mean tilting factor (-)',fontsize=20)
        fig.tight_layout()
        fig.savefig('plot/m_ed.png')
        plt.close(fig)

    # plot
    if 'contact_distrib_m_ed' in dict_user['L_figures']:
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,9))
        # mean
        ax1.plot(dict_user['L_m_ed_contact'], label='Contact')
        ax1.plot(dict_user['L_m_ed_large_contact'], label= 'Large contact')
        ax1.legend()
        ax1.set_title('Mean (-)',fontsize=20)
        # distribution
        ax2.plot(dict_user['L_m_ed_plus_contact'], label='+ Contact')
        ax2.plot(dict_user['L_m_ed_minus_contact'], label='- Contact')
        ax2.plot(dict_user['L_m_ed_plus_large_contact'], label= '+ Large contact')
        ax2.plot(dict_user['L_m_ed_minus_large_contact'], label= '- Large contact')
        ax2.legend()
        ax2.set_title('Mean of + and - (-)')
        # close
        plt.suptitle('Mean tilting factor in contact (-)',fontsize=20)
        fig.tight_layout()
        fig.savefig('plot/contact_distrib_m_ed.png')
        plt.close(fig)

    # find nearest node to contact point
    with open('data/dem_to_main.data', 'rb') as handle:
        dict_save = pickle.load(handle)
    contactPoint = dict_save['contact_point']
    L_search = list(abs(np.array(dict_sample['x_L']-contactPoint[0])))
    i_x = L_search.index(min(L_search))
    L_search = list(abs(np.array(dict_sample['y_L']-contactPoint[1])))
    i_y = L_search.index(min(L_search))
    # tracker
    dict_user['L_ed_contact_point'].append(ed_map[-1-i_y, i_x])
    # plot
    if 'contact_point_ed' in dict_user['L_figures']:
        fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
        ax1.plot(dict_user['L_ed_contact_point'])
        ax1.set_title('Tilting factor at contact point (-)',fontsize=20)
        fig.tight_layout()
        fig.savefig('plot/contact_point_ed.png')
        plt.close(fig)

# -----------------------------------------------------------------------------#

def add_element_list(data_list, data_sum, data_n):
    '''
    Add element to a list with condition.

    if data_n != 0, add data_sum/data_n
    else, add 0
    '''
    if data_n != 0:
        data_list.append(data_sum/data_n)
    else : 
        data_list.append(0)

# -----------------------------------------------------------------------------#

def compute_dt_PF_Aitken(dict_user, dict_sample):
    '''
    Compute the time step used in PF simulation with a method inspired by Aitken.

    Several threshold values are used.
    '''
    # level 0
    if abs(dict_user['L_m_ed_contact'][-1]) < dict_user['Aitken_1']:
        dict_user['dt_PF'] = dict_user['dt_PF_0']
    # level 1
    if dict_user['Aitken_1'] <= abs(dict_user['L_m_ed_contact'][-1]) and abs(dict_user['L_m_ed_contact'][-1]) < dict_user['Aitken_2']:
        dict_user['dt_PF'] = dict_user['dt_PF_1']
    # level 2
    if dict_user['Aitken_2'] <= abs(dict_user['L_m_ed_contact'][-1]) :
        dict_user['dt_PF'] = dict_user['dt_PF_2']
    
    # save and plot
    dict_user['L_dt_PF'].append(dict_user['dt_PF'])
    if 'dt_PF' in dict_user['L_figures']:
        fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
        ax1.plot(dict_user['L_dt_PF'])
        ax1.set_yticks([dict_user['dt_PF_0'], dict_user['dt_PF_1'], dict_user['dt_PF_2']])
        ax1.set_yticklabels(['Level 0', 'Level 1', 'Level 2'])
        ax1.set_title('Time step used for PF (s)',fontsize=20)
        fig.tight_layout()
        fig.savefig('plot/dt_PF.png')
        plt.close(fig)

#-------------------------------------------------------------------------------

def write_eta_txt(dict_user, dict_sample):
    '''
    Write a .txt file needed for MOOSE simulation.

    This .txt file represent the phase field maps.
    '''
    file_to_write_1 = open('data/eta_1.txt','w')
    file_to_write_2 = open('data/eta_2.txt','w')
    # x
    file_to_write_1.write('AXIS X\n')
    file_to_write_2.write('AXIS X\n')
    line = ''
    for x in dict_sample['x_L']:
        line = line + str(x)+ ' '
    line = line + '\n'
    file_to_write_1.write(line)
    file_to_write_2.write(line)
    # y
    file_to_write_1.write('AXIS Y\n')
    file_to_write_2.write('AXIS Y\n')
    line = ''
    for y in dict_sample['y_L']:
        line = line + str(y)+ ' '
    line = line + '\n'
    file_to_write_1.write(line)
    file_to_write_2.write(line)
    # data
    file_to_write_1.write('DATA\n')
    file_to_write_2.write('DATA\n')
    for j in range(len(dict_sample['y_L'])):
        for i in range(len(dict_sample['x_L'])):
            # grain 1
            file_to_write_1.write(str(dict_sample['eta_1_map'][-1-j,i])+'\n')
            # grain 2
            file_to_write_2.write(str(dict_sample['eta_2_map'][-1-j,i])+'\n')
    # close
    file_to_write_1.close()
    file_to_write_2.close()

#-------------------------------------------------------------------------------

def write_c_txt(dict_user, dict_sample):
    '''
    Write a .txt file needed for MOOSE simulation.

    This .txt represents the solute map.
    '''
    file_to_write = open('data/c.txt','w')
    # x
    file_to_write.write('AXIS X\n')
    line = ''
    for x in dict_sample['x_L']:
        line = line + str(x)+ ' '
    line = line + '\n'
    file_to_write.write(line)
    # y
    file_to_write.write('AXIS Y\n')
    line = ''
    for y in dict_sample['y_L']:
        line = line + str(y)+ ' '
    line = line + '\n'
    file_to_write.write(line)
    # data
    file_to_write.write('DATA\n')
    for j in range(len(dict_sample['y_L'])):
        for i in range(len(dict_sample['x_L'])):
            file_to_write.write(str(dict_sample['c_map'][-1-j,i])+'\n')
    # close
    file_to_write.close()

#-------------------------------------------------------------------------------

def write_as_txt(dict_user, dict_sample):
    '''
    Write a .txt file needed for MOOSE simulation.

    This .txt represents the map of the solid activity.
    '''
    file_to_write = open('data/as.txt','w')
    # x
    file_to_write.write('AXIS X\n')
    line = ''
    for x in dict_sample['x_L']:
        line = line + str(x)+ ' '
    line = line + '\n'
    file_to_write.write(line)
    # y
    file_to_write.write('AXIS Y\n')
    line = ''
    for y in dict_sample['y_L']:
        line = line + str(y)+ ' '
    line = line + '\n'
    file_to_write.write(line)
    # data
    file_to_write.write('DATA\n')
    for j in range(len(dict_sample['y_L'])):
        for i in range(len(dict_sample['x_L'])):
            file_to_write.write(str(dict_sample['as_map'][-1-j,i])+'\n')
    # close
    file_to_write.close()

#-------------------------------------------------------------------------------

def compute_kc(dict_user, dict_sample):
    '''
    Compute the diffusion coefficient of the solute.
    Then Write a .txt file needed for MOOSE simulation.

    This .txt file represent the phase field maps.
    '''
    # loading old variable
    c_map = dict_sample['c_map']
    # updating solute map
    c_map_new = c_map 

    # compute
    kc_map = np.array(np.zeros((dict_user['n_mesh_y'], dict_user['n_mesh_x'])), dtype = bool)
    # iterate on x and y
    for i_y in range(len(dict_sample['y_L'])):
        # fast checking
        if max(dict_sample['eta_1_map'][i_y, :])<0.5 and max(dict_sample['eta_2_map'][i_y, :])<0.5:
            kc_map[i_y, :] = True
        # individual checking
        else :
            for i_x in range(len(dict_sample['x_L'])):
                if dict_sample['eta_1_map'][i_y, i_x] < 0.5 and dict_sample['eta_2_map'][i_y, i_x] < 0.5: # out of the grain
                    kc_map[i_y, i_x] = True
                elif dict_sample['eta_1_map'][i_y, i_x] > 0.5 and dict_sample['eta_2_map'][i_y, i_x] > 0.5: # in the contact
                    kc_map[i_y, i_x] = True
                else :
                    kc_map[i_y, i_x] = False

    # dilation
    dilated_M = binary_dilation(kc_map, dict_user['struct_element'])

    #compute the map of the solute diffusion coefficient
    kc_map = dict_user['D_solute']*dilated_M

    # write
    file_to_write_1 = open('data/kc.txt','w')
    # x
    file_to_write_1.write('AXIS X\n')
    line = ''
    for x in dict_sample['x_L']:
        line = line + str(x)+ ' '
    line = line + '\n'
    file_to_write_1.write(line)
    # y
    file_to_write_1.write('AXIS Y\n')
    line = ''
    for y in dict_sample['y_L']:
        line = line + str(y)+ ' '
    line = line + '\n'
    file_to_write_1.write(line)
    # data
    file_to_write_1.write('DATA\n')
    for j in range(len(dict_sample['y_L'])):
        for i in range(len(dict_sample['x_L'])):
            # grain 1
            file_to_write_1.write(str(kc_map[-1-j,i])+'\n')
    # close
    file_to_write_1.close()

    # compute the number of grain detected in kc_map
    invert_dilated_M = np.invert(dilated_M)
    labelled_image, num_features = label(invert_dilated_M)
    dict_user['L_grain_kc_map'].append(num_features)

    # plot 
    if 'n_grain_kc_map' in dict_user['L_figures']:
        fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
        ax1.plot(dict_user['L_grain_kc_map'])
        ax1.set_title('Number of grains detected (-)',fontsize=20)
        fig.tight_layout()
        fig.savefig('plot/n_grain_detected.png')
        plt.close(fig)

    # iterate on the mesh
    for i_y in range(len(dict_sample['y_L'])):
        for i_x in range(len(dict_sample['x_L'])):
            # push solute out of the solid
            if not dilated_M[i_y, i_x] and c_map[i_y, i_x] > 1: # threshold value
                solute_moved = False
                size_window = 1
                # compute solute to move
                solute_to_move = c_map[i_y, i_x] - 1
                while not solute_moved :
                    i_window = 0
                    while not solute_moved and i_window <= size_window:
                        n_node_available = 0

                        #Look to move horizontaly and vertically
                        if i_window == 0 :
                            top_available = False
                            down_available = False
                            left_available = False
                            right_available = False
                            #to the top
                            if i_y - size_window > 0:
                                top_available = dilated_M[i_y-size_window, i_x]
                                if dilated_M[i_y-size_window, i_x] :
                                    n_node_available = n_node_available + 1
                            #to the down
                            if i_y + size_window < len(dict_sample['y_L']):
                                down_available = dilated_M[i_y+size_window, i_x]
                                if dilated_M[i_y+size_window, i_x] :
                                    n_node_available = n_node_available + 1
                            #to the left
                            if i_x - size_window > 0:
                                left_available = dilated_M[i_y, i_x-size_window]
                                if dilated_M[i_y, i_x-size_window] :
                                    n_node_available = n_node_available + 1
                            #to the right
                            if i_x + size_window < len(dict_sample['x_L']):
                                right_available = dilated_M[i_y, i_x+size_window]
                                if dilated_M[i_y, i_x+size_window] :
                                    n_node_available = n_node_available + 1

                            #move solute if et least one node is available
                            if n_node_available != 0 :
                                #to the top
                                if top_available:
                                    c_map_new[i_y-size_window, i_x] = c_map_new[i_y-size_window, i_x] + solute_to_move/n_node_available
                                #to the down
                                if down_available:
                                    c_map_new[i_y+size_window, i_x] = c_map_new[i_y+size_window, i_x] + solute_to_move/n_node_available
                                #to the left
                                if left_available:
                                    c_map_new[i_y, i_x-size_window] = c_map_new[i_y, i_x-size_window] + solute_to_move/n_node_available
                                #to the right
                                if right_available:
                                    c_map_new[i_y, i_x+size_window] = c_map_new[i_y, i_x+size_window] + solute_to_move/n_node_available
                                c_map_new[i_y, i_x] = 1
                                solute_moved = True

                        #Look to move diagonally
                        else :
                            top_min_available = False
                            top_max_available = False
                            down_min_available = False
                            down_max_available = False
                            left_min_available = False
                            left_max_available = False
                            right_min_available = False
                            right_max_available = False
                            #to the top
                            if i_y - size_window > 0:
                                if i_x - i_window > 0 :
                                    top_min_available = dilated_M[i_y-size_window, i_x-i_window]
                                    if dilated_M[i_y-size_window, i_x-i_window] :
                                        n_node_available = n_node_available + 1
                                if i_x + i_window < len(dict_sample['x_L']):
                                    top_max_available = dilated_M[i_y-size_window, i_x+i_window]
                                    if dilated_M[i_y-size_window, i_x+i_window] :
                                        n_node_available = n_node_available + 1
                            #to the down
                            if i_y + size_window < len(dict_sample['y_L']):
                                if i_x - i_window > 0 :
                                    down_min_available = dilated_M[i_y+size_window, i_x-i_window]
                                    if dilated_M[i_y+size_window, i_x-i_window] :
                                        n_node_available = n_node_available + 1
                                if i_x + i_window < len(dict_sample['x_L']):
                                    down_max_available = dilated_M[i_y+size_window, i_x+i_window]
                                    if dilated_M[i_y+size_window, i_x+i_window] :
                                        n_node_available = n_node_available + 1
                            #to the left
                            if i_x - size_window > 0:
                                if i_y - i_window > 0 :
                                    left_min_available = dilated_M[i_y-i_window, i_x-size_window]
                                    if dilated_M[i_y-i_window, i_x-size_window] :
                                        n_node_available = n_node_available + 1
                                if i_y + i_window < len(dict_sample['y_L']):
                                    left_max_available = dilated_M[i_y+i_window, i_x-size_window]
                                    if dilated_M[i_y+i_window, i_x-size_window] :
                                        n_node_available = n_node_available + 1
                            #to the right
                            if i_x + size_window < len(dict_sample['x_L']):
                                if i_x - i_window > 0 :
                                    right_min_available = dilated_M[i_y-i_window, i_x+size_window]
                                    if dilated_M[i_y-i_window, i_x+size_window] :
                                        n_node_available = n_node_available + 1
                                if i_y + i_window < len(dict_sample['y_L']):
                                    right_max_available = dilated_M[i_y+i_window, i_x+size_window]
                                    if dilated_M[i_y+i_window, i_x+size_window] :
                                        n_node_available = n_node_available + 1

                            #move solute if et least one node is available
                            if n_node_available != 0 :
                                #to the top
                                if top_min_available:
                                    c_map_new[i_y-size_window, i_x-i_window] = c_map_new[i_y-size_window, i_x-i_window] + solute_to_move/n_node_available
                                if top_max_available:
                                    c_map_new[i_y-size_window, i_x+i_window] = c_map_new[i_y-size_window, i_x+i_window] + solute_to_move/n_node_available
                                #to the down
                                if down_min_available:
                                    c_map_new[i_y+size_window, i_x-i_window] = c_map_new[i_y+size_window, i_x-i_window] + solute_to_move/n_node_available
                                if down_max_available:
                                    c_map_new[i_y+size_window, i_x+i_window] = c_map_new[i_y+size_window, i_x+i_window] + solute_to_move/n_node_available
                                #to the left
                                if left_min_available:
                                    c_map_new[i_y-i_window, i_x-size_window] = c_map_new[i_y-i_window, i_x-size_window] + solute_to_move/n_node_available
                                if left_max_available:
                                    c_map_new[i_y+i_window, i_x-size_window] = c_map_new[i_y+i_window, i_x-size_window] + solute_to_move/n_node_available
                                #to the right
                                if right_min_available:
                                    c_map_new[i_y-i_window, i_x+size_window] = c_map_new[i_y-i_window, i_x+size_window] + solute_to_move/n_node_available
                                if right_max_available:
                                    c_map_new[i_y+i_window, i_x+size_window] = c_map_new[i_y+i_window, i_x+size_window] + solute_to_move/n_node_available
                                c_map_new[i_y, i_x] = 1
                                solute_moved = True
                        i_window = i_window + 1
                    size_window = size_window + 1   

            # push solute in of the solid
            if not dilated_M[i_y, i_x] and c_map[i_y, i_x] < 1: # threshold value
                solute_moved = False
                size_window = 1
                # compute solute to move
                solute_to_move = 1 - c_map[i_y, i_x]
                while not solute_moved :
                    i_window = 0
                    while not solute_moved and i_window <= size_window:
                        n_node_available = 0

                        #Look to move horizontaly and vertically
                        if i_window == 0 :
                            top_available = False
                            down_available = False
                            left_available = False
                            right_available = False
                            #to the top
                            if i_y - size_window > 0:
                                top_available = dilated_M[i_y-size_window, i_x]
                                if dilated_M[i_y-size_window, i_x] :
                                    n_node_available = n_node_available + 1
                            #to the down
                            if i_y + size_window < len(dict_sample['y_L']):
                                down_available = dilated_M[i_y+size_window, i_x]
                                if dilated_M[i_y+size_window, i_x] :
                                    n_node_available = n_node_available + 1
                            #to the left
                            if i_x - size_window > 0:
                                left_available = dilated_M[i_y, i_x-size_window]
                                if dilated_M[i_y, i_x-size_window] :
                                    n_node_available = n_node_available + 1
                            #to the right
                            if i_x + size_window < len(dict_sample['x_L']):
                                right_available = dilated_M[i_y, i_x+size_window]
                                if dilated_M[i_y, i_x+size_window] :
                                    n_node_available = n_node_available + 1

                            #move solute if et least one node is available
                            if n_node_available != 0 :
                                #to the top
                                if top_available:
                                    c_map_new[i_y-size_window, i_x] = c_map_new[i_y-size_window, i_x] - solute_to_move/n_node_available
                                #to the down
                                if down_available:
                                    c_map_new[i_y+size_window, i_x] = c_map_new[i_y+size_window, i_x] - solute_to_move/n_node_available
                                #to the left
                                if left_available:
                                    c_map_new[i_y, i_x-size_window] = c_map_new[i_y, i_x-size_window] - solute_to_move/n_node_available
                                #to the right
                                if right_available:
                                    c_map_new[i_y, i_x+size_window] = c_map_new[i_y, i_x+size_window] - solute_to_move/n_node_available
                                c_map_new[i_y, i_x] = 1
                                solute_moved = True

                        #Look to move diagonally
                        else :
                            top_min_available = False
                            top_max_available = False
                            down_min_available = False
                            down_max_available = False
                            left_min_available = False
                            left_max_available = False
                            right_min_available = False
                            right_max_available = False
                            #to the top
                            if i_y - size_window > 0:
                                if i_x - i_window > 0 :
                                    top_min_available = dilated_M[i_y-size_window, i_x-i_window]
                                    if dilated_M[i_y-size_window, i_x-i_window] :
                                        n_node_available = n_node_available + 1
                                if i_x + i_window < len(dict_sample['x_L']):
                                    top_max_available = dilated_M[i_y-size_window, i_x+i_window]
                                    if dilated_M[i_y-size_window, i_x+i_window] :
                                        n_node_available = n_node_available + 1
                            #to the down
                            if i_y + size_window < len(dict_sample['y_L']):
                                if i_x - i_window > 0 :
                                    down_min_available = dilated_M[i_y+size_window, i_x-i_window]
                                    if dilated_M[i_y+size_window, i_x-i_window] :
                                        n_node_available = n_node_available + 1
                                if i_x + i_window < len(dict_sample['x_L']):
                                    down_max_available = dilated_M[i_y+size_window, i_x+i_window]
                                    if dilated_M[i_y+size_window, i_x+i_window] :
                                        n_node_available = n_node_available + 1
                            #to the left
                            if i_x - size_window > 0:
                                if i_y - i_window > 0 :
                                    left_min_available = dilated_M[i_y-i_window, i_x-size_window]
                                    if dilated_M[i_y-i_window, i_x-size_window] :
                                        n_node_available = n_node_available + 1
                                if i_y + i_window < len(dict_sample['y_L']):
                                    left_max_available = dilated_M[i_y+i_window, i_x-size_window]
                                    if dilated_M[i_y+i_window, i_x-size_window] :
                                        n_node_available = n_node_available + 1
                            #to the right
                            if i_x + size_window < len(dict_sample['x_L']):
                                if i_x - i_window > 0 :
                                    right_min_available = dilated_M[i_y-i_window, i_x+size_window]
                                    if dilated_M[i_y-i_window, i_x+size_window] :
                                        n_node_available = n_node_available + 1
                                if i_y + i_window < len(dict_sample['y_L']):
                                    right_max_available = dilated_M[i_y+i_window, i_x+size_window]
                                    if dilated_M[i_y+i_window, i_x+size_window] :
                                        n_node_available = n_node_available + 1

                            #move solute if et least one node is available
                            if n_node_available != 0 :
                                #to the top
                                if top_min_available:
                                    c_map_new[i_y-size_window, i_x-i_window] = c_map_new[i_y-size_window, i_x-i_window] - solute_to_move/n_node_available
                                if top_max_available:
                                    c_map_new[i_y-size_window, i_x+i_window] = c_map_new[i_y-size_window, i_x+i_window] - solute_to_move/n_node_available
                                #to the down
                                if down_min_available:
                                    c_map_new[i_y+size_window, i_x-i_window] = c_map_new[i_y+size_window, i_x-i_window] - solute_to_move/n_node_available
                                if down_max_available:
                                    c_map_new[i_y+size_window, i_x+i_window] = c_map_new[i_y+size_window, i_x+i_window] - solute_to_move/n_node_available
                                #to the left
                                if left_min_available:
                                    c_map_new[i_y-i_window, i_x-size_window] = c_map_new[i_y-i_window, i_x-size_window] - solute_to_move/n_node_available
                                if left_max_available:
                                    c_map_new[i_y+i_window, i_x-size_window] = c_map_new[i_y+i_window, i_x-size_window] - solute_to_move/n_node_available
                                #to the right
                                if right_min_available:
                                    c_map_new[i_y-i_window, i_x+size_window] = c_map_new[i_y-i_window, i_x+size_window] - solute_to_move/n_node_available
                                if right_max_available:
                                    c_map_new[i_y+i_window, i_x+size_window] = c_map_new[i_y+i_window, i_x+size_window] - solute_to_move/n_node_available
                                c_map_new[i_y, i_x] = 1
                                solute_moved = True
                        i_window = i_window + 1
                    size_window = size_window + 1   

    # save data
    dict_sample['c_map'] = c_map_new

    # write txt for the solute concentration map
    write_c_txt(dict_user, dict_sample) # solute

#-------------------------------------------------------------------------------

def write_i(dict_user, dict_sample):
  '''
  Create the .i file to run MOOSE simulation.

  The file is generated from a template nammed PF_ACS_base.i
  '''
  file_to_write = open('pf.i','w')
  file_to_read = open('pf_base.i','r')
  lines = file_to_read.readlines()
  file_to_read.close()

  j = 0
  for line in lines :
    j = j + 1
    if j == 4:
      line = line[:-1] + ' ' + str(len(dict_sample['x_L'])-1)+'\n'
    elif j == 5:
      line = line[:-1] + ' ' + str(len(dict_sample['y_L'])-1)+'\n'
    elif j == 6:
      line = line[:-1] + ' ' + str(min(dict_sample['x_L']))+'\n'
    elif j == 7:
      line = line[:-1] + ' ' + str(max(dict_sample['x_L']))+'\n'
    elif j == 8:
      line = line[:-1] + ' ' + str(min(dict_sample['y_L']))+'\n'
    elif j == 9:
      line = line[:-1] + ' ' + str(max(dict_sample['y_L']))+'\n'
    elif j == 116:
      line = line[:-1] + "'"+str(dict_user['Mobility_eff'])+' '+str(dict_user['kappa_eta'])+" 1'\n"
    elif j == 138:
      line = line[:-1] + ' ' + str(dict_user['Energy_barrier'])+"'\n"
    elif j == 151:
      line = line[:-1] + "'" + str(1*(dict_sample['x_L'][1]-dict_sample['x_L'][0]))\
                       + ' 1 ' + str(dict_user['k_diss']) + ' ' + str(dict_user['k_prec']) + ' '\
                       + str(1*(dict_sample['x_L'][1]-dict_sample['x_L'][0])*(dict_sample['y_L'][1]-dict_sample['y_L'][0])) + " 1'\n"
    elif j == 219:
      line = line[:-1] + ' ' + str(dict_user['dt_PF']*dict_user['n_t_PF']) +'\n'
    elif j == 223:
      line = line[:-1] + ' ' + str(dict_user['dt_PF']) +'\n'
    file_to_write.write(line)

  file_to_write.close()

# -----------------------------------------------------------------------------#

def sort_files_yade():
  '''
  Sort vtk files from yade simulation.
  '''
  # look for the next indice
  j = 0
  filepath = Path('vtk/grains_'+str(j)+'.vtk')
  while filepath.exists():
      j = j + 1
      filepath = Path('vtk/grains_'+str(j)+'.vtk')
  # rename
  os.rename('vtk/grains-polyhedra-00000000.vtk','vtk/grains_'+str(j)+'.vtk')
  os.rename('vtk/grains-polyhedra-00000001.vtk','vtk/grains_'+str(j+1)+'.vtk')

# -----------------------------------------------------------------------------#

def compute_contact_volume(dict_user, dict_sample):
    '''
    Characterize the contact volume.
    '''
    # build a polyhedra of the contact volume
    L_vertices = interpolate_vertices_contact(dict_sample['eta_1_map'], dict_sample['eta_2_map'], dict_user, dict_sample)
    # adapt for plot and search box sizes
    L_vertices_x = []
    L_vertices_y = []
    min_set = False # bool for the first iteration
    for v in L_vertices:
        L_vertices_x.append(v[0])
        L_vertices_y.append(v[1])
        # set min, max values with the first vertex
        if not min_set :
            min_x = v[0]
            max_x = v[0]
            min_y = v[1]
            max_y = v[1]
            min_set = True
        # compare to find extremum
        else :
            if v[0] < min_x:
                min_x = v[0]
            if max_x < v[0]:
                max_x = v[0]
            if v[1] < min_y:
                min_y = v[1]
            if max_y < v[1]:
                max_y = v[1]
    L_vertices_x.append(L_vertices_x[0])
    L_vertices_y.append(L_vertices_y[0])

    # save box contact
    dict_sample['box_contact_x_min'] = min_x
    dict_sample['box_contact_x_max'] = max_x
    dict_sample['box_contact_y_min'] = min_y
    dict_sample['box_contact_y_max'] = max_y

    # capturing the grains boundaries
    L_vertices_1 = interpolate_vertices(dict_sample['eta_1_map'], dict_sample['pos_1'], dict_user, dict_sample) # from pf_to_dem.py
    L_vertices_2 = interpolate_vertices(dict_sample['eta_2_map'], dict_sample['pos_2'], dict_user, dict_sample) # from pf_to_dem.py

    # plot
    if 'contact_detection' in dict_user['L_figures']:
        fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
        # g1
        L_x, L_y = tuplet_to_list_no_centerized(L_vertices_1) # from tools.py
        ax1.plot(L_x, L_y, label='G1')
        # g2
        L_x, L_y = tuplet_to_list_no_centerized(L_vertices_2) # from tools.py
        ax1.plot(L_x, L_y, label='G2')
        # contact 
        #ax1.plot(L_vertices_x, L_vertices_y, 'x', label='Contact')
        # box contact
        ax1.plot([min_x, max_x, max_x, min_x, min_x], [min_y, min_y, max_y, max_y, min_y], label='Contact Box')
        # close
        ax1.legend()
        ax1.axis('equal')
        plt.suptitle('Contact Detection', fontsize=20)
        fig.tight_layout()
        if dict_user['print_all_contact_detection']:
            fig.savefig('plot/contact_detection/'+str(dict_sample['i_DEMPF_ite'])+'.png')
        else:
            fig.savefig('plot/contact_detection.png')
        plt.close(fig)

# -----------------------------------------------------------------------------#

def interpolate_vertices_contact(eta_1_map, eta_2_map, dict_user, dict_sample):
    '''
    Interpolate vertices for a contact between two polyhedrons.
    '''
    L_vertices = []
    for i_x in range(len(dict_sample['x_L'])-1):
        for i_y in range(len(dict_sample['y_L'])-1):
            L_in_1 = [] # list the nodes inside the grain 1
            if eta_1_map[-1-i_y    , i_x] > dict_user['eta_contact_box_detection'] :
                L_in_1.append(0)
            if eta_1_map[-1-(i_y+1), i_x] > dict_user['eta_contact_box_detection'] :
                L_in_1.append(1)
            if eta_1_map[-1-(i_y+1), i_x+1] > dict_user['eta_contact_box_detection'] :
                L_in_1.append(2)
            if eta_1_map[-1-i_y    , i_x+1] > dict_user['eta_contact_box_detection'] :
                L_in_1.append(3)
            m_eta_1 = (eta_1_map[-1-i_y, i_x]+eta_1_map[-1-(i_y+1), i_x]+eta_1_map[-1-(i_y+1), i_x+1]+eta_1_map[-1-i_y, i_x+1])/4
            L_in_2 = [] # list the nodes inside the grain 2
            if eta_2_map[-1-i_y    , i_x] > dict_user['eta_contact_box_detection'] :
                L_in_2.append(0)
            if eta_2_map[-1-(i_y+1), i_x] > dict_user['eta_contact_box_detection'] :
                L_in_2.append(1)
            if eta_2_map[-1-(i_y+1), i_x+1] > dict_user['eta_contact_box_detection'] :
                L_in_2.append(2)
            if eta_2_map[-1-i_y    , i_x+1] > dict_user['eta_contact_box_detection'] :
                L_in_2.append(3)
            m_eta_2 = (eta_2_map[-1-i_y, i_x]+eta_2_map[-1-(i_y+1), i_x]+eta_2_map[-1-(i_y+1), i_x+1]+eta_2_map[-1-i_y, i_x+1])/4
            
            if (L_in_1 != [] and L_in_1 != [0,1,2,3]) and (m_eta_2>dict_user['eta_contact_box_detection']):
                # iterate on the lines of the mesh to find the plane intersection for grain 1
                L_p_1 = []
                if (0 in L_in_1 and 1 not in L_in_1) or (0 not in L_in_1 and 1 in L_in_1):# line 01
                    x_p = dict_sample['x_L'][i_x]
                    y_p = (dict_user['eta_contact_box_detection']-eta_1_map[-1-i_y, i_x])/(eta_1_map[-1-(i_y+1), i_x]-eta_1_map[-1-i_y, i_x])*(dict_sample['y_L'][i_y+1]-dict_sample['y_L'][i_y])+dict_sample['y_L'][i_y]
                    L_p_1.append(np.array([x_p, y_p]))
                if (1 in L_in_1 and 2 not in L_in_1) or (1 not in L_in_1 and 2 in L_in_1):# line 12
                    x_p = (dict_user['eta_contact_box_detection']-eta_1_map[-1-(i_y+1), i_x])/(eta_1_map[-1-(i_y+1), i_x+1]-eta_1_map[-1-(i_y+1), i_x])*(dict_sample['x_L'][i_x+1]-dict_sample['x_L'][i_x])+dict_sample['x_L'][i_x]
                    y_p = dict_sample['y_L'][i_y+1]
                    L_p_1.append(np.array([x_p, y_p]))
                if (2 in L_in_1 and 3 not in L_in_1) or (2 not in L_in_1 and 3 in L_in_1):# line 23
                    x_p = dict_sample['x_L'][i_x+1]
                    y_p = (dict_user['eta_contact_box_detection']-eta_1_map[-1-i_y, i_x+1])/(eta_1_map[-1-(i_y+1), i_x+1]-eta_1_map[-1-i_y, i_x+1])*(dict_sample['y_L'][i_y+1]-dict_sample['y_L'][i_y])+dict_sample['y_L'][i_y]
                    L_p_1.append(np.array([x_p, y_p]))
                if (3 in L_in_1 and 0 not in L_in_1) or (3 not in L_in_1 and 0 in L_in_1):# line 30
                    x_p = (dict_user['eta_contact_box_detection']-eta_1_map[-1-i_y, i_x])/(eta_1_map[-1-i_y, i_x+1]-eta_1_map[-1-i_y, i_x])*(dict_sample['x_L'][i_x+1]-dict_sample['x_L'][i_x])+dict_sample['x_L'][i_x]
                    y_p = dict_sample['y_L'][i_y]
                    L_p_1.append(np.array([x_p, y_p]))
                # compute the mean point
                p_mean_1 = np.array([0,0])
                for p in L_p_1 :
                    p_mean_1 = p_mean_1 + p
                p_mean_1 = p_mean_1/len(L_p_1)
                p_mean_1_b = True
            else :
                p_mean_1_b = False

            if (m_eta_1>dict_user['eta_contact_box_detection']) and (L_in_2 != [] and L_in_2 != [0,1,2,3]):
                # iterate on the lines of the mesh to find the plane intersection for grain 2
                L_p_2 = []
                if (0 in L_in_2 and 1 not in L_in_2) or (0 not in L_in_2 and 1 in L_in_2):# line 01
                    x_p = dict_sample['x_L'][i_x]
                    y_p = (dict_user['eta_contact_box_detection']-eta_2_map[-1-i_y, i_x])/(eta_2_map[-1-(i_y+1), i_x]-eta_2_map[-1-i_y, i_x])*(dict_sample['y_L'][i_y+1]-dict_sample['y_L'][i_y])+dict_sample['y_L'][i_y]
                    L_p_2.append(np.array([x_p, y_p]))
                if (1 in L_in_2 and 2 not in L_in_2) or (1 not in L_in_2 and 2 in L_in_2):# line 12
                    x_p = (dict_user['eta_contact_box_detection']-eta_2_map[-1-(i_y+1), i_x])/(eta_2_map[-1-(i_y+1), i_x+1]-eta_2_map[-1-(i_y+1), i_x])*(dict_sample['x_L'][i_x+1]-dict_sample['x_L'][i_x])+dict_sample['x_L'][i_x]
                    y_p = dict_sample['y_L'][i_y+1]
                    L_p_2.append(np.array([x_p, y_p]))
                if (2 in L_in_2 and 3 not in L_in_2) or (2 not in L_in_2 and 3 in L_in_2):# line 23
                    x_p = dict_sample['x_L'][i_x+1]
                    y_p = (dict_user['eta_contact_box_detection']-eta_2_map[-1-i_y, i_x+1])/(eta_2_map[-1-(i_y+1), i_x+1]-eta_2_map[-1-i_y, i_x+1])*(dict_sample['y_L'][i_y+1]-dict_sample['y_L'][i_y])+dict_sample['y_L'][i_y]
                    L_p_2.append(np.array([x_p, y_p]))
                if (3 in L_in_2 and 0 not in L_in_2) or (3 not in L_in_2 and 0 in L_in_2):# line 30
                    x_p = (dict_user['eta_contact_box_detection']-eta_2_map[-1-i_y, i_x])/(eta_2_map[-1-i_y, i_x+1]-eta_2_map[-1-i_y, i_x])*(dict_sample['x_L'][i_x+1]-dict_sample['x_L'][i_x])+dict_sample['x_L'][i_x]
                    y_p = dict_sample['y_L'][i_y]
                    L_p_2.append(np.array([x_p, y_p]))
                # compute the mean point
                p_mean_2 = np.array([0,0])
                for p in L_p_2 :
                    p_mean_2 = p_mean_2 + p
                p_mean_2 = p_mean_2/len(L_p_2)
                p_mean_2_b = True
            else :
                p_mean_2_b = False
                
            if p_mean_1_b and not p_mean_2_b :
                L_vertices.append(p_mean_1)
            if not p_mean_1_b and p_mean_2_b :
                L_vertices.append(p_mean_2)
            elif p_mean_1_b and p_mean_2_b :
                # compute a commun vertex
                L_vertices.append((p_mean_1+p_mean_2)/2)

    return L_vertices

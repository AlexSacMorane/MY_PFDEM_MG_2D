# -*- encoding=utf-8 -*-

import pickle, math, os, shutil, random
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
    print('Updating phase field maps')

    # load data
    with open('data/dem_to_main.data', 'rb') as handle:
        dict_save = pickle.load(handle)
    L_displacement = dict_save['L_displacement']

    # tracker
    dict_user['L_L_displacement'].append(dict_save['L_displacement'])
    if 'displacement' in dict_user['L_figures']:
        plot_displacement(dict_user, dict_sample) # from tools.py

    # iterate on grains
    for i_grain in range(0, len(dict_sample['L_etai_map'])):
        # read displacement
        displacement = L_displacement[i_grain] 
        # print
        #print('grain', i_grain, ':', displacement)

        # TRANSLATION on x
        # loading old variables
        eta_i_map = dict_sample['L_etai_map'][i_grain]
        # updating phase map
        eta_i_map_new = np.zeros((dict_user['n_mesh_y'], dict_user['n_mesh_x']))
        # iteration on x
        for i_x in range(len(dict_sample['x_L'])):
            x = dict_sample['x_L'][i_x]
            i_x_old = 0
            # eta 1 
            if displacement[0] < 0:
                if x-displacement[0] <= dict_sample['x_L'][-1]:
                    # look for window
                    while not (dict_sample['x_L'][i_x_old] <= x-displacement[0] and x-displacement[0] <= dict_sample['x_L'][i_x_old+1]):
                        i_x_old = i_x_old + 1
                    # interpolate
                    eta_i_map_new[:, i_x] = (eta_i_map[:, (i_x_old+1)] - eta_i_map[:, i_x_old])/(dict_sample['x_L'][i_x_old+1] - dict_sample['x_L'][i_x_old])*\
                                                (x-displacement[0] - dict_sample['x_L'][i_x_old]) + eta_i_map[:, i_x_old]
            elif displacement[0] > 0:
                if dict_sample['x_L'][0] <= x-displacement[0]:
                    # look for window
                    while not (dict_sample['x_L'][i_x_old] <= x-displacement[0] and x-displacement[0] <= dict_sample['x_L'][i_x_old+1]):
                        i_x_old = i_x_old + 1
                    # interpolate
                    eta_i_map_new[:, i_x] = (eta_i_map[:, (i_x_old+1)] - eta_i_map[:, i_x_old])/(dict_sample['x_L'][i_x_old+1] - dict_sample['x_L'][i_x_old])*\
                                                (x-displacement[0] - dict_sample['x_L'][i_x_old]) + eta_i_map[:, i_x_old]
            else :
                eta_i_map_new = eta_i_map
        
        # TRANSLATION on y
        # loading old variables
        eta_i_map = eta_i_map_new.copy()
        # updating phase map
        eta_i_map_new = np.zeros((dict_user['n_mesh_y'], dict_user['n_mesh_x']))
        # iteration on y
        for i_y in range(len(dict_sample['y_L'])):
            y = dict_sample['y_L'][i_y]
            i_y_old = 0
            # eta 1 
            if displacement[1] < 0:
                if y-displacement[1] <= dict_sample['y_L'][-1]:
                    # look for window
                    while not (dict_sample['y_L'][i_y_old] <= y-displacement[1] and y-displacement[1] <= dict_sample['y_L'][i_y_old+1]):
                        i_y_old = i_y_old + 1
                    # interpolate
                    eta_i_map_new[-1-i_y, :] = (eta_i_map[-1-(i_y_old+1), :] - eta_i_map[-1-i_y_old, :])/(dict_sample['y_L'][i_y_old+1] - dict_sample['y_L'][i_y_old])*\
                                                (y-displacement[1] - dict_sample['y_L'][i_y_old]) + eta_i_map[-1-i_y_old, :]
            elif displacement[1] > 0:
                if dict_sample['y_L'][0] <= y-displacement[1]:
                    # look for window
                    while not (dict_sample['y_L'][i_y_old] <= y-displacement[1] and y-displacement[1] <= dict_sample['y_L'][i_y_old+1]):
                        i_y_old = i_y_old + 1
                    # interpolate
                    eta_i_map_new[-1-i_y, :] = (eta_i_map[-1-(i_y_old+1), :] - eta_i_map[-1-i_y_old, :])/(dict_sample['y_L'][i_y_old+1] - dict_sample['y_L'][i_y_old])*\
                                                (y-displacement[1] - dict_sample['y_L'][i_y_old]) + eta_i_map[-1-i_y_old, :]
            else :
                eta_i_map_new = eta_i_map

        # ROTATION
        if displacement[2] != 0:
            # compute center of mass
            center_x = 0
            center_y = 0
            center_z = 0
            counter = 0
            # iterate on x
            for i_x in range(len(dict_sample['x_L'])):
                # iterate on y
                for i_y in range(len(dict_sample['y_L'])):
                    # criterion to verify the point is inside the grain
                    if dict_user['eta_contact_box_detection'] < eta_i_map_new[-1-i_y, i_x]:
                        center_x = center_x + dict_sample['x_L'][i_x]
                        center_y = center_y + dict_sample['y_L'][i_y]
                        counter = counter + 1
            # compute the center of mass
            center_x = center_x/counter
            center_y = center_y/counter
            center = np.array([center_x, center_y, 0])
                        
            # compute matrice of rotation
            # cf french wikipedia "quaternions et rotation dans l'espace"
            a = displacement[2]
            b = displacement[3]
            c = displacement[4]
            d = displacement[5]
            M_rot = np.array([[a*a+b*b-c*c-d*d,     2*b*c-2*a*d,     2*a*c+2*b*d],
                                [    2*a*d+2*b*c, a*a-b*b+c*c-d*d,     2*c*d-2*a*b],
                                [    2*b*d-2*a*c,     2*a*b+2*c*d, a*a-b*b-c*c+d*d]])
            M_rot_inv = np.linalg.inv(M_rot)
            
            # loading old variables
            eta_i_map = eta_i_map_new.copy()
            # updating phase map
            eta_i_map_new = np.zeros((dict_user['n_mesh_y'], dict_user['n_mesh_x']))
            # iteration on x
            for i_x in range(len(dict_sample['x_L'])):
                # iteration on y
                for i_y in range(len(dict_sample['y_L'])):
                    # create vector of the node
                    p = np.array([dict_sample['x_L'][i_x], dict_sample['y_L'][i_y], 0])
                    # remove the center of the grain
                    pp = p - center
                    # applied the invert rotation
                    pp = np.dot(M_rot_inv, pp)
                    # applied center
                    pp = pp + center
                    # initialization 
                    found = True
                    # look for the vector in the x-axis                    
                    if dict_sample['x_L'][0] <= pp[0] and pp[0] <= dict_sample['x_L'][-1]:
                        i_x_old = 0
                        while not (dict_sample['x_L'][i_x_old] <= pp[0] and pp[0] <= dict_sample['x_L'][i_x_old+1]):
                            i_x_old = i_x_old + 1
                    else :
                        found = False
                    # look for the vector in the y-axis                    
                    if dict_sample['y_L'][0] <= pp[1] and pp[1] <= dict_sample['y_L'][-1]:
                        i_y_old = 0
                        while not (dict_sample['y_L'][i_y_old] <= pp[1] and pp[1] <= dict_sample['y_L'][i_y_old+1]):
                            i_y_old = i_y_old + 1
                    else :
                        found = False
                    # double interpolation if point found
                    if found :
                        # interpolation following the z-axis
                        # points
                        p00 = np.array([  dict_sample['x_L'][i_x_old],   dict_sample['y_L'][i_y_old]])
                        p10 = np.array([dict_sample['x_L'][i_x_old+1],   dict_sample['y_L'][i_y_old]])
                        p01 = np.array([  dict_sample['x_L'][i_x_old], dict_sample['y_L'][i_y_old+1]])
                        p11 = np.array([dict_sample['x_L'][i_x_old+1], dict_sample['y_L'][i_y_old+1]])
                        # values
                        q00 = eta_i_map[-1-i_y_old    ,   i_x_old]
                        q10 = eta_i_map[-1-i_y_old    , i_x_old+1]
                        q01 = eta_i_map[-1-(i_y_old+1),   i_x_old]
                        q11 = eta_i_map[-1-(i_y_old+1), i_x_old+1]
                        
                        # interpolation following the y-axis
                        # points
                        p0 = np.array([  dict_sample['x_L'][i_x_old]])
                        p1 = np.array([dict_sample['x_L'][i_x_old+1]])
                        # values
                        q0 = (q00*(p01[1]-pp[1]) + q01*(pp[1]-p00[1]))/(p01[1]-p00[1])
                        q1 = (q10*(p11[1]-pp[1]) + q11*(pp[1]-p10[1]))/(p11[1]-p10[1])

                        # interpolation following the x-axis
                        eta_i_map_new[-1-i_y, i_x] = (q0*(p1[0]-pp[0]) + q1*(pp[0]-p0[0]))/(p1[0]-p0[0])
                        
                    else :
                        eta_i_map_new[-1-i_y, i_x] = 0
                              
        # update variables
        dict_sample['L_etai_map'][i_grain] = eta_i_map_new

# -----------------------------------------------------------------------------#

def compute_contact(dict_user, dict_sample):
    '''
    Compute the contact characteristics:
        - box
        - maximum surface
        - volume
    '''
    # load data
    with open('data/dem_to_main.data', 'rb') as handle:
        dict_save = pickle.load(handle)
    L_contact = dict_save['L_contact']
    # initialization
    dict_sample['L_vol_contact'] = []
    dict_sample['L_surf_contact'] = []

    # iterate on contacts
    for contact in L_contact:   
        # volume
        vol_contact = 0
        # points in contact
        L_points_contact = []
        # iterate on mesh
        for i_x in range(len(dict_sample['x_L'])):
            for i_y in range(len(dict_sample['y_L'])):
                # contact detection
                if dict_sample['L_etai_map'][contact[0]][-1-i_y, i_x] > dict_user['eta_contact_box_detection'] and\
                    dict_sample['L_etai_map'][contact[1]][-1-i_y, i_x] > dict_user['eta_contact_box_detection']:
        
                    # points in contact
                    L_points_contact.append(np.array([dict_sample['x_L'][i_x], dict_sample['y_L'][i_y]]))
            
                    # compute volume contact
                    vol_contact = vol_contact + (dict_sample['x_L'][1]-dict_sample['x_L'][0])*\
                                                (dict_sample['y_L'][1]-dict_sample['y_L'][0])
             
        # compute surface
        surf_contact = 0
        for i in range(len(L_points_contact)-1):
           for j in range(i+1, len(L_points_contact)):
                # compute vector
                u = L_points_contact[i]-L_points_contact[j]
                v = contact[3]/np.linalg.norm(contact[3])
                v = np.array([-v[1], v[0]])
                # look for larger surface
                if abs(np.dot(u,v))> surf_contact:
                   surf_contact = abs(np.dot(u,v))

        # save
        dict_sample['L_vol_contact'].append(vol_contact)
        dict_sample['L_surf_contact'].append(surf_contact)

    # save
    # iterate on potential contact
    ij = 0
    for i_grain in range(len(dict_sample['L_etai_map'])-1):
        for j_grain in range(i_grain+1, len(dict_sample['L_etai_map'])):
            if len(L_contact)>0:
                i_contact = 0
                contact_found = L_contact[i_contact][0:2] == [i_grain, j_grain]
                while not contact_found and i_contact < len(L_contact)-1:
                        i_contact = i_contact + 1
                        contact_found = L_contact[i_contact][0:2] == [i_grain, j_grain]
            else :
                contact_found = False
            if dict_sample['i_DEMPF_ite'] == 1:
                if contact_found:
                    dict_user['L_L_contact_volume'].append([dict_sample['L_vol_contact'][i_contact]])
                    dict_user['L_L_contact_surface'].append([dict_sample['L_surf_contact'][i_contact]])
                else:
                    dict_user['L_L_contact_volume'].append([0])
                    dict_user['L_L_contact_surface'].append([0])
            else :
                if contact_found:
                    dict_user['L_L_contact_volume'][ij].append(dict_sample['L_vol_contact'][i_contact])
                    dict_user['L_L_contact_surface'][ij].append(dict_sample['L_surf_contact'][i_contact])
                else:
                    dict_user['L_L_contact_volume'][ij].append(0)
                    dict_user['L_L_contact_surface'][ij].append(0)
            ij = ij + 1

# -----------------------------------------------------------------------------#

def compute_as(dict_user, dict_sample):
    '''
    Compute activity of solid.
    '''
    # load data
    with open('data/dem_to_main.data', 'rb') as handle:
        dict_save = pickle.load(handle)
    L_contact = dict_save['L_contact']

    # init
    dict_sample['as_map'] = np.ones((dict_user['n_mesh_y'], dict_user['n_mesh_x']))
    L_pressure_tempo = []
    L_as_tempo = []

    # iterate on contacts
    for i_contact in range(len(L_contact)):
        p_saved = False
        contact = L_contact[i_contact]
        # iterate on mesh
        for i_x in range(len(dict_sample['x_L'])):
            for i_y in range(len(dict_sample['y_L'])):
                # contact detection
                if dict_sample['L_etai_map'][contact[0]][-1-i_y, i_x] > dict_user['eta_contact_box_detection'] and\
                    dict_sample['L_etai_map'][contact[1]][-1-i_y, i_x] > dict_user['eta_contact_box_detection']:
                    # determine pressure
                    P = np.linalg.norm(contact[3])/dict_sample['L_surf_contact'][i_contact] # Pa
                    # tempo save
                    if not p_saved :
                        L_pressure_tempo.append(P)
                        L_as_tempo.append(math.exp(P*dict_user['V_m']/(dict_user['R_cst']*dict_user['temperature'])))
                        p_saved = True
                    # save in the map
                    # do not erase data
                    if dict_sample['as_map'][-1-i_y, i_x] == 1:
                        dict_sample['as_map'][-1-i_y, i_x] = math.exp(P*dict_user['V_m']/(dict_user['R_cst']*dict_user['temperature']))

    # save
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
                    dict_user['L_L_contact_pressure'].append([L_pressure_tempo[i_contact]])
                    dict_user['L_L_contact_as'].append([L_as_tempo[i_contact]])
                else:
                    dict_user['L_L_contact_pressure'].append([0])
                    dict_user['L_L_contact_as'].append([1])
            else :
                if contact_found:
                    dict_user['L_L_contact_pressure'][ij].append(L_pressure_tempo[i_contact])
                    dict_user['L_L_contact_as'][ij].append(L_as_tempo[i_contact])
                else:
                    dict_user['L_L_contact_pressure'][ij].append(0)
                    dict_user['L_L_contact_as'][ij].append(1)
            ij = ij + 1
            
    # plot 
    plot_as_pressure(dict_user, dict_sample) # from tools.py

    # write as
    write_array_txt(dict_sample, 'as', dict_sample['as_map'])

#-------------------------------------------------------------------------------

def compute_kc(dict_user, dict_sample):
    '''
    Compute the diffusion coefficient of the solute.
    Then write a .txt file needed for MOOSE simulation.

    This .txt file represent the diffusion coefficient map.
    '''
    # compute
    kc_map = np.array(np.zeros((dict_user['n_mesh_y'], dict_user['n_mesh_x'])), dtype = bool)
    kc_pore_map =  np.array(np.zeros((dict_user['n_mesh_y'], dict_user['n_mesh_x'])), dtype = bool)
    # iterate on x and y 
    for i_y in range(len(dict_sample['y_L'])):
        for i_x in range(len(dict_sample['x_L'])):
            # count the number of eta > eta_criterion
            c_eta_crit = 0
            for i_grain in range(len(dict_sample['L_etai_map'])):
                if dict_sample['L_etai_map'][i_grain][i_y, i_x] > dict_user['eta_contact_box_detection']:
                    c_eta_crit = c_eta_crit + 1
            # compute coefficient of diffusion
            if c_eta_crit == 0: # out of the grain
                kc_map[i_y, i_x] = True
                kc_pore_map[i_y, i_x] = True
            elif c_eta_crit >= 2: # in the contact
                kc_map[i_y, i_x] = True
                kc_pore_map[i_y, i_x] = False
            else : # in the grain
                kc_map[i_y, i_x] = False
                kc_pore_map[i_y, i_x] = False

    # dilation
    dilated_M = binary_dilation(kc_map, dict_user['struct_element'])

    #compute the map of the solute diffusion coefficient
    kc_map = dict_user['D_solute']*dilated_M + 99*dict_user['D_solute']*kc_pore_map
    dict_sample['kc_map'] = kc_map

    # write
    write_array_txt(dict_sample, 'kc', dict_sample['kc_map'])

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

    # loading old variable
    c_map = dict_sample['c_map']
    # updating solute map
    c_map_new = c_map.copy()

    # iterate on the mesh
    for i_y in range(len(dict_sample['y_L'])):
        for i_x in range(len(dict_sample['x_L'])):
            if not dilated_M[i_y, i_x]: 
                c_map_new[i_y, i_x] = dict_user['C_eq']
    
    # HERE MUST BE MODIFIED
    # Move the solute to connserve the mass
                    
    # save data
    dict_sample['c_map'] = c_map_new

    # write txt for the solute concentration map
    write_array_txt(dict_sample, 'c', dict_sample['c_map'])

#-------------------------------------------------------------------------------

def write_array_txt(dict_sample, namefile, data_array):
    '''
    Write a .txt file needed for MOOSE simulation.

    This .txt represents the map of a numpy array.
    '''
    file_to_write = open('data/'+namefile+'.txt','w')
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
            file_to_write.write(str(data_array[-1-j, i])+'\n')
    # close
    file_to_write.close()

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
    elif j == 14:
      line = ''
      for i_grain in range(len(dict_sample['L_phii_map'])):
        line = line + '\t[./eta'+str(i_grain+1)+']\n'+\
                      '\t\torder = FIRST\n'+\
                      '\t\tfamily = LAGRANGE\n'+\
                      '\t\t[./InitialCondition]\n'+\
                      '\t\t\ttype = FunctionIC\n'+\
                      '\t\t\tfunction = eta'+str(i_grain+1)+'_txt\n'+\
                      '\t\t[../]\n'+\
                      '\t[../]\n'
    elif j == 24:
      line = ''
      for i_grain in range(len(dict_sample['L_phii_map'])):
        line = line + '\t# Order parameter eta'+str(i_grain+1)+'\n'+\
                      '\t[./deta'+str(i_grain+1)+'dt]\n'+\
                      '\t\ttype = TimeDerivative\n'+\
                      '\t\tvariable = eta'+str(i_grain+1)+'\n'+\
                      '\t[../]\n'+\
                      '\t[./ACBulk'+str(i_grain+1)+']\n'+\
                      '\t\ttype = AllenCahn\n'+\
                      '\t\tvariable = eta'+str(i_grain+1)+'\n'+\
                      '\t\tmob_name = L\n'+\
                      '\t\tf_name = F_total\n'+\
                      '\t[../]\n'+\
                      '\t[./ACInterface'+str(i_grain+1)+']\n'+\
                      '\t\ttype = ACInterface\n'+\
                      '\t\tvariable = eta'+str(i_grain+1)+'\n'+\
                      '\t\tmob_name = L\n'+\
                      "\t\tkappa_name = 'kappa_eta'\n"+\
                      '\t[../]\n'
    elif j == 30:
      line = ''
      for i_grain in range(len(dict_sample['L_phii_map'])):
        line = line + '\t[./eta'+str(i_grain+1)+'_c]\n'+\
                      '\t\ttype = CoefCoupledTimeDerivative\n'+\
                      "\t\tv = 'eta"+str(i_grain+1)+"'\n"+\
                      '\t\tvariable = c\n'+\
                      '\t\tcoef = '+str(1/dict_user['V_m'])+'\n'+\
                      '\t[../]\n'    
    elif j == 43:
      line = line[:-1] + "'"+str(dict_user['Mobility_eff'])+' '+str(dict_user['kappa_eta'])+" 1'\n"
    elif j == 63:
      line = line[:-1] + "'"
      for i_grain in range(len(dict_sample['L_phii_map'])):
        line = line +'eta'+str(i_grain+1)+' '
      line = line[:-1] + "'\n"
    elif j == 65:
      line = line[:-1] + "'" + str(dict_user['Energy_barrier'])+"'\n"
    elif j == 66:
      line = line[:-1] + "'"
      for i_grain in range(len(dict_sample['L_phii_map'])):
        line = line +'h*(eta'+str(i_grain+1)+'^2*(1-eta'+str(i_grain+1)+')^2)+'
      line = line[:-1] + "'\n"
    elif j == 75:
      line = line[:-1] + "'"
      for i_grain in range(len(dict_sample['L_phii_map'])):
        line = line +'eta'+str(i_grain+1)+' '
      line = line + "c'\n"
    elif j == 78:
      line = line[:-1] + "'" + str(dict_user['C_eq']) + ' ' + str(dict_user['k_diss']) + ' ' + str(dict_user['k_prec']) + "'\n"
    elif j == 79:
      line = line[:-1] + "'if(c<c_eq*as,k_diss,k_prec)*as*(1-c/(c_eq*as))*("
      for i_grain in range(len(dict_sample['L_phii_map'])):
        line = line +'3*eta'+str(i_grain+1)+'^2-2*eta'+str(i_grain+1)+'^3+'
      line = line[:-1] +")'\n"
    elif j == 88:
      line = line[:-1] + "'"
      for i_grain in range(len(dict_sample['L_phii_map'])):
        line = line +'eta'+str(i_grain+1)+' '
      line = line + "c'\n"
    elif j == 89:
      line = line[:-1] + "'F("
      for i_grain in range(len(dict_sample['L_phii_map'])):
        line = line +'eta'+str(i_grain+1)+','
      line = line[:-1] + ") Ed("
      for i_grain in range(len(dict_sample['L_phii_map'])):
        line = line +'eta'+str(i_grain+1)+','
      line = line + "c)'\n"
    elif j == 98:
      line = ''
      for i_grain in range(len(dict_sample['L_phii_map'])):
        line = line + '\t[eta'+str(i_grain+1)+'_txt]\n'+\
                      '\t\ttype = PiecewiseMultilinear\n'+\
                      "\t\tdata_file = data/eta_"+str(i_grain+1)+".txt\n"+\
                      '\t[]\n'   
    elif j == 132 or j == 133 or j == 135 or j == 136:
      line = line[:-1] + ' ' + str(dict_user['crit_res']) +'\n'
    elif j == 139:
      line = line[:-1] + ' ' + str(dict_user['dt_PF']*dict_user['n_t_PF']) +'\n'
    elif j == 143:
      line = line[:-1] + ' ' + str(dict_user['dt_PF']) +'\n'
    file_to_write.write(line)

  file_to_write.close()
    
#-------------------------------------------------------------------------------

def sort_dem_files(dict_user, dict_sample):
    '''
    Sort the files from the YADE simulation.
    '''
    # rename files
    os.rename('vtk/ite_PFDEM_'+str(dict_sample['i_DEMPF_ite'])+'_lsBodies.0.vtm',\
                'vtk/ite_PFDEM_'+str(dict_sample['i_DEMPF_ite'])+'.vtm')

#-------------------------------------------------------------------------------

def sort_phase_variable(dict_user, dict_sample):
    '''
    Reduce the number of phase variables used by combining them.
    '''
    # preparation of the lists
    L_i_etaj_neighbor = []
    for i_eta in range(len(dict_sample['L_etai_map'])):
        L_i_etaj_neighbor.append([])
    # preparation of the dilation
    binary_structure = np.ones((4,4))

    # detect etas neighbor
    for i_eta in range(len(dict_sample['L_etai_map'])-1):
        # compute mask of i_eta
        mask_i = dict_sample['L_etai_map'][i_eta].copy() > dict_user['eta_contact_box_detection']
        # dilatex
        mask_i = binary_dilation(mask_i, binary_structure)
        # iterate on others etas
        for j_eta in range(i_eta+1, len(dict_sample['L_etai_map'])):
            # compute mask of j_eta
            mask_j = dict_sample['L_etai_map'][j_eta].copy() > dict_user['eta_contact_box_detection']
            # dilate
            mask_j = binary_dilation(mask_j, binary_structure)
            # iteration on the mesh
            neighbors = False
            i_x = 0
            while not neighbors and i_x < dict_user['n_mesh_x']:
                i_y = 0 
                while not neighbors and i_y < dict_user['n_mesh_y']:
                    if mask_i[i_y, i_x] and mask_j[i_y, i_x] :
                        neighbors = True
                    i_y = i_y + 1
                i_x = i_x + 1
            # save
            if neighbors:
                L_i_etaj_neighbor[i_eta].append(j_eta)
                L_i_etaj_neighbor[j_eta].append(i_eta)
    
    # sort etas in phis
    L_phi_L_etas = [[]]
    L_phi_neighbors = [[]]
    L_random_etai = list(range(len(dict_sample['L_etai_map'])))
    random.shuffle(L_random_etai)

    for eta_i in L_random_etai:
        included = False
        # iteration on phis
        phi_i = 0
        while not included and phi_i < len(L_phi_L_etas):
            if eta_i not in L_phi_neighbors[phi_i]:
                included = True
                L_phi_L_etas[phi_i].append(eta_i)
                for neighbor in L_i_etaj_neighbor[eta_i]:
                    if neighbor not in L_phi_neighbors[phi_i]:
                        L_phi_neighbors[phi_i].append(neighbor)
            phi_i = phi_i + 1
        # creation of a new phi
        if not included:
            L_phi_L_etas.append([eta_i])
            L_phi_neighbors.append(L_i_etaj_neighbor[eta_i])
    
    # compute phi maps and box
    L_phi = []
    L_phi_L_boxs = []
    for phi_i in range(len(L_phi_L_etas)):
        # initialization
        phi_i_map = np.zeros((dict_user['n_mesh_y'], dict_user['n_mesh_x']))
        L_boxs = []
        # iteration on etas of this phi
        for eta_i in L_phi_L_etas[phi_i]:
            # sum etas
            phi_i_map = phi_i_map + dict_sample['L_etai_map'][eta_i]    
            
            # look for box
            # -x limit
            i_x = 0
            found = False
            while (not found) and (i_x < dict_sample['L_etai_map'][eta_i].shape[1]):
                if np.max(dict_sample['L_etai_map'][eta_i][:, i_x]) < dict_user['eta_contact_box_detection']:
                    i_x_min = i_x
                else :
                    found = True
                i_x = i_x + 1
            # +x limit
            i_x = dict_sample['L_etai_map'][eta_i].shape[1]-1
            found = False
            while not found and 0 <= i_x:
                if np.max(dict_sample['L_etai_map'][eta_i][:, i_x]) < dict_user['eta_contact_box_detection']:
                    i_x_max = i_x
                else :
                    found = True
                i_x = i_x - 1
            # -y limit
            i_y = 0
            found = False
            while (not found) and (i_y < dict_sample['L_etai_map'][eta_i].shape[0]):
                if np.max(dict_sample['L_etai_map'][eta_i][i_y, :]) < dict_user['eta_contact_box_detection']:
                    i_y_min = i_y
                else :
                    found = True
                i_y = i_y + 1
            # +y limit
            i_y = dict_sample['L_etai_map'][eta_i].shape[0]-1
            found = False
            while not found and 0 <= i_y:
                if np.max(dict_sample['L_etai_map'][eta_i][i_y, :]) < dict_user['eta_contact_box_detection']:
                    i_y_max = i_y
                else :
                    found = True
                i_y = i_y - 1            
            # save
            L_boxs.append([i_x_min, i_x_max, i_y_min, i_y_max])
        # save
        L_phi.append(phi_i_map)
        L_phi_L_boxs.append(L_boxs)

    # save
    dict_sample['L_phi_L_etas'] = L_phi_L_etas   
    dict_sample['L_phi_L_boxs'] = L_phi_L_boxs   
    dict_sample['L_phii_map'] = L_phi       

    # plot
    for i_phi in range(len(dict_sample['L_phii_map'])):
        fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
        im = ax1.imshow(dict_sample['L_phii_map'][i_phi], interpolation = 'nearest', extent=(dict_sample['x_L'][0],dict_sample['x_L'][-1],dict_sample['y_L'][0],dict_sample['y_L'][-1]))
        fig.colorbar(im, ax=ax1)
        ax1.set_title(r'$\phi$'+str(i_phi),fontsize = 30)
        fig.tight_layout()
        fig.savefig('plot/phi_'+str(i_phi)+'.png')
        plt.close(fig)

        

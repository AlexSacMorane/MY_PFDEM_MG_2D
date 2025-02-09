# -*- encoding=utf-8 -*-

import pickle, math, os, shutil, random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import vtk, skfmm
from vtk.util.numpy_support import vtk_to_numpy
from scipy.ndimage import label, binary_dilation

# own
import tools

# -----------------------------------------------------------------------------#

def sort_files(dict_user, dict_sample):
    '''
    Sort files generated by MOOSE to different directories
    '''
    os.rename('pf_out.e','vtk/pf_out.e')
    os.rename('pf.i','input/pf.i')
    j = 0
    j_str = index_to_str(j)
    j_total_str = index_to_str(dict_user['j_total'])
    filepath = Path('pf_other_'+j_str+'.pvtu')
    while filepath.exists():
        for i_proc in range(dict_user['n_proc']):
            os.rename('pf_other_'+j_str+'_'+str(i_proc)+'.vtu','vtk/pf_other_'+j_str+'_'+str(i_proc)+'.vtu')
            shutil.copyfile('vtk/pf_other_'+j_str+'_'+str(i_proc)+'.vtu','vtk/pf_'+j_total_str+'_'+str(i_proc)+'.vtu')
        os.rename('pf_other_'+j_str+'.pvtu','vtk/pf_other_'+j_str+'.pvtu')
        # write .pvtu to save all vtk
        file = open('vtk/pf_'+j_total_str+'.pvtu','w')
        file.write('''<?xml version="1.0"?>
        <VTKFile type="PUnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
        \t<PUnstructuredGrid GhostLevel="1">
        \t\t<PPointData>
        \t\t\t<PDataArray type="Float64" Name="as"/>
        \t\t\t<PDataArray type="Float64" Name="kc"/>\n''')
        for i_phi in range(len(dict_sample['L_phii_map'])):
            file.write('''\t\t\t\t<PDataArray type="Float64" Name="eta'''+str(i_phi+1)+'''"/>\n''')
        file.write('''\t\t\t\t<PDataArray type="Float64" Name="c"/>
        \t\t</PPointData>
        \t\t<PCellData>
        \t\t\t<PDataArray type="Int32" Name="libmesh_elem_id"/>
        \t\t\t<PDataArray type="Int32" Name="subdomain_id"/>
        \t\t\t<PDataArray type="Int32" Name="processor_id"/>
        \t\t</PCellData>
        \t\t<PPoints>
        \t\t\t<PDataArray type="Float64" Name="Points" NumberOfComponents="3"/>
        \t\t</PPoints>''')
        line = ''
        for i_proc in range(dict_user['n_proc']):
            line = line + '''\t\t<Piece Source="pf_'''+j_total_str+'''_'''+str(i_proc)+'''.vtu"/>\n'''
        file.write(line)
        file.write('''\t</PUnstructuredGrid>
        </VTKFile>''')
        file.close()
        j = j + 1
        j_str = index_to_str(j)
        filepath = Path('pf_other_'+j_str+'.pvtu')
        dict_user['j_total'] = dict_user['j_total'] + 1
        j_total_str = index_to_str(dict_user['j_total'])
    return index_to_str(j-1)

# -----------------------------------------------------------------------------#

def index_to_str(j):
    '''
    Convert a integer into a string with the format XXX.
    '''
    if j < 10:
        return '00'+str(j)
    elif 10<=j and j<100:
        return '0'+str(j)
    else :
        return str(j)

# -----------------------------------------------------------------------------#

def read_vtk(dict_user, dict_sample, j_str):
    '''
    Read the last vtk files to obtain data from MOOSE.
    '''
    if not dict_sample['Map_known']:
        L_XYZ = []
        L_L_i_XY = []

    # iterate on the proccessors used
    for i_proc in range(dict_user['n_proc']):
        print('processor',i_proc+1,'/',dict_user['n_proc'])

        # name of the file to load
        namefile = 'vtk/pf_other_'+j_str+'_'+str(i_proc)+'.vtu'

        # load a vtk file as input
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(namefile)
        reader.Update()

        # Grab a scalar from the vtk file
        nodes_vtk_array = reader.GetOutput().GetPoints().GetData()
        L_etai_vtk_array = []
        for i_grain in range(len(dict_sample['L_phii_map'])):
            L_etai_vtk_array.append(reader.GetOutput().GetPointData().GetArray("eta"+str(i_grain+1)))
        c_vtk_array = reader.GetOutput().GetPointData().GetArray("c")

        #Get the coordinates of the nodes and the scalar values
        nodes_array = vtk_to_numpy(nodes_vtk_array)
        L_etai_array = []
        for i_grain in range(len(dict_sample['L_phii_map'])):
            L_etai_array.append(vtk_to_numpy(L_etai_vtk_array[i_grain]))
        c_array = vtk_to_numpy(c_vtk_array)

        # map is not know
        if not dict_sample['Map_known']:
            # save the map
            L_i_XY = []
            # Must detect common zones between processors
            for i_XYZ in range(len(nodes_array)) :
                XYZ = nodes_array[i_XYZ]
                # Do not consider twice a point
                if list(XYZ) not in L_XYZ :
                    # search node in the mesh
                    L_search = list(abs(np.array(dict_sample['x_L']-list(XYZ)[0])))
                    i_x = L_search.index(min(L_search))
                    L_search = list(abs(np.array(dict_sample['y_L']-list(XYZ)[1])))
                    i_y = L_search.index(min(L_search))
                    # save map
                    L_XYZ.append(list(XYZ))
                    L_i_XY.append([i_x, i_y])
                    # rewrite map
                    for i_grain in range(len(dict_sample['L_phii_map'])):
                        dict_sample['L_phii_map'][i_grain][-1-i_y, i_x] = L_etai_array[i_grain][i_XYZ]
                    dict_sample['c_map'][-1-i_y, i_x] = c_array[i_XYZ]
                else :
                    L_i_XY.append(None)
            # Here the algorithm can be help as the mapping is known
            L_L_i_XY.append(L_i_XY)

        # map is known
        else :
            # iterate on data
            for i_XYZ in range(len(nodes_array)) :
                # read
                if dict_sample['L_L_i_XY_used'][i_proc][i_XYZ] != None :
                    i_x = dict_sample['L_L_i_XY_used'][i_proc][i_XYZ][0]
                    i_y = dict_sample['L_L_i_XY_used'][i_proc][i_XYZ][1]
                    # rewrite map
                    for i_grain in range(len(dict_sample['L_phii_map'])):
                        dict_sample['L_phii_map'][i_grain][-1-i_y, i_x] = L_etai_array[i_grain][i_XYZ]
                    dict_sample['c_map'][-1-i_y, i_x] = c_array[i_XYZ]
    
    if not dict_sample['Map_known']:
        # the map is known
        dict_sample['Map_known'] = True
        dict_sample['L_L_i_XY_used'] = L_L_i_XY

# -----------------------------------------------------------------------------#

def compute_levelset(dict_user, dict_sample):
    '''
    From a phase map, compute level set function.
    '''
    L_sdf_i_map = []
    L_x_L_i = []
    L_y_L_i = []
    L_rbm_to_apply = []
    # iterate on the phase variable
    for i_grain in range(len(dict_sample['L_etai_map'])):
        # compute binary map
        bin_i_map = np.array(-np.ones((dict_user['n_mesh_y'], dict_user['n_mesh_x'])))
        
        # iteration on x
        for i_x in range(len(dict_sample['x_L'])):
            # iteration on y
            for i_y in range(len(dict_sample['y_L'])):
                # grain 
                if dict_sample['L_etai_map'][i_grain][-1-i_y, i_x] > 0.5:
                    bin_i_map[-1-i_y, i_x] = 1
        
        
        # look for dimensions of box    
        # -x limit
        i_x = 0
        found = False
        while (not found) and (i_x < bin_i_map.shape[0]):
            if np.max(bin_i_map[:, i_x]) == -1:
                i_x_min_lim = i_x
            else :
                found = True
            i_x = i_x + 1
        # +x limit
        i_x = bin_i_map.shape[1]-1
        found = False
        while not found and 0 <= i_x:
            if np.max(bin_i_map[:, i_x]) == -1:
                i_x_max_lim = i_x
            else :
                found = True
            i_x = i_x - 1
        # number of nodes on x
        n_nodes_x = i_x_max_lim-i_x_min_lim+1
        # -y limit
        i_y = 0
        found = False
        while not found and 0 <= i_y:
            if np.max(bin_i_map[-1-i_y, :]) == -1:
                i_y_min_lim = i_y
            else :
                found = True
            i_y = i_y + 1
        # +y limit
        i_y = bin_i_map.shape[0]-1
        found = False
        while (not found) and (i_y < bin_i_map.shape[1]):
            if np.max(bin_i_map[-1-i_y, :]) == -1:
                i_y_max_lim = i_y
            else :
                found = True
            i_y = i_y - 1
        # number of nodes on y
        n_nodes_y = i_y_max_lim-i_y_min_lim+1

        # extraction of data
        bin_i_map = bin_i_map[-1-i_y_max_lim:-1-i_y_min_lim+1,
                              i_x_min_lim:i_x_max_lim+1] 
        
        # adaptation map
        bin_i_map_adapt = -np.ones((bin_i_map.shape[1], bin_i_map.shape[0], dict_user['extrude_z']))
        for i_y in range(bin_i_map.shape[0]):
            for i_x in range(bin_i_map.shape[1]):
                for i_z in range(dict_user['margins'], dict_user['extrude_z']-dict_user['margins']):
                    bin_i_map_adapt[i_x, i_y, i_z] = bin_i_map[-1-i_y, i_x]
         
        # creation of sub mesh
        m_size = dict_user['m_size']
        x_L_i = np.arange(-m_size*(n_nodes_x-1)/2,
                           m_size*(n_nodes_x-1)/2+0.1*m_size,
                           m_size)
        y_L_i = np.arange(-m_size*(n_nodes_y-1)/2,
                           m_size*(n_nodes_y-1)/2+0.1*m_size,
                           m_size)

        # compute rbm to apply
        rbm_to_apply = [dict_sample['x_L'][i_x_min_lim]-x_L_i[0],
                        dict_sample['y_L'][i_y_min_lim]-y_L_i[0],
                        0]

        # compute signed distance function
        sdf_i_map = -skfmm.distance(bin_i_map_adapt, dx=np.array([m_size, m_size, m_size]))

        # save
        L_sdf_i_map.append(sdf_i_map)
        L_x_L_i.append(x_L_i)
        L_y_L_i.append(y_L_i)
        L_rbm_to_apply.append(rbm_to_apply)

    # save data
    dict_save = {
    'L_sdf_i_map': L_sdf_i_map,
    'L_rbm_to_apply': L_rbm_to_apply,
    'L_x_L_i': L_x_L_i,
    'L_y_L_i': L_y_L_i
    }
    with open('data/level_set.data', 'wb') as handle:
        pickle.dump(dict_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

# -----------------------------------------------------------------------------#

def extract_etas_from_phis(dict_user, dict_sample):
    '''
    Extract etas (grain description) from the phis (moose outputs).
    '''
    # iterate on eta
    for eta_i in range(len(dict_sample['L_etai_map'])):
        # look for phi and position
        phi = 0
        while eta_i not in dict_sample['L_phi_L_etas'][phi]:
            phi = phi + 1
        # extract boxs
        box = dict_sample['L_phi_L_boxs'][phi][dict_sample['L_phi_L_etas'][phi].index(eta_i)]
        # extend box
        delta_x = box[1]-box[0]
        i_x_min = max(0, box[1]-int(1.1*delta_x))
        i_x_max = min(dict_user['n_mesh_x']-1, box[0]+int(1.1*delta_x))
        delta_y = box[3]-box[2]
        i_y_min = max(0, box[3]-int(1.1*delta_y))
        i_y_max = min(dict_user['n_mesh_y']-1, box[2]+int(1.1*delta_y))

        # segmentation
        map_extract = dict_sample['L_phii_map'][phi][i_y_min:i_y_max+1, i_x_min:i_x_max+1].copy()
        for i in range(map_extract.shape[0]):
            for j in range(map_extract.shape[1]):
                if map_extract[i, j] > dict_user['eta_contact_box_detection']:
                    map_extract[i, j] = 1
                else :
                    map_extract[i, j] = 0
        labelled_image, num_features = label(map_extract)
        # at least two features
        if num_features > 1:
            # determine the largest domain
            L_surf = [0]*num_features
            for i in range(map_extract.shape[0]):
                for j in range(map_extract.shape[1]):
                    if labelled_image[i, j] != 0:
                        L_surf[int(labelled_image[i,j]-1)] = L_surf[int(labelled_image[i,j]-1)] + 1
            # compute the mask
            mask_map = labelled_image == (L_surf.index(max(L_surf))+1)
        else :
            mask_map = np.ones(map_extract.shape)
        # dilate mask
        binary_structure = np.ones((2,2))
        mask_map = binary_dilation(mask_map, binary_structure)

        # write phase variables
        for i_x in range(i_x_min, i_x_max+1):
            for i_y in range(i_y_min, i_y_max+1):
                dict_sample['L_etai_map'][eta_i][i_y, i_x] = dict_sample['L_phii_map'][phi][i_y, i_x]*mask_map[i_y-i_y_min, i_x-i_x_min]
  
# -----------------------------------------------------------------------------#

def check_etas(dict_user, dict_sample):
    '''
    Check the integrity of the phase variables etas.

    If the phase variable is not needed, it is deleted from the simulation (?).
    '''
    # iterate on the phase variables
    eta_i = 0
    while eta_i < len(dict_sample['L_etai_map']):
        deleted = False
        # compute bin
        bin_map = dict_sample['L_etai_map'][eta_i].copy() > 0.5
        # check if the variable is not needed 
        if np.sum(bin_map) == 0 :
            deleted = True
            # delete in conjugate trackers
            ij = 0
            L_ij_tempo = []
            for i_grain in range(len(dict_sample['L_etai_map'])-1):
                for j_grain in range(i_grain+1, len(dict_sample['L_etai_map'])):
                    if i_grain == eta_i or j_grain == eta_i: 
                        L_ij_tempo.append(ij)  
                    ij = ij + 1
            L_ij_tempo.reverse()
            for ij in L_ij_tempo:
                dict_user['L_L_contact_volume'].pop(ij)
                dict_user['L_L_contact_surface'].pop(ij)
                dict_user['L_L_contact_pressure'].pop(ij)
                dict_user['L_L_contact_as'].pop(ij)
                dict_user['L_L_overlap'].pop(ij)
                dict_user['L_L_normal_force'].pop(ij)
            # delete in simple trackers
            dict_user['L_L_sum_eta_i'].pop(eta_i)
            dict_user['L_L_m_eta_i'].pop(eta_i)
            dict_user['L_L_loss_move_pf_eta_i'].pop(eta_i)
            dict_user['L_L_loss_kc_eta_i'].pop(eta_i)
            dict_user['L_L_loss_pf_eta_i'].pop(eta_i)
            # delete map
            dict_sample['L_etai_map'].pop(eta_i)
            # move initial shape in the tracker
            m_eta_i = dict_user['L_initial_eta'].pop(eta_i)
            dict_user['L_initial_eta'].append(m_eta_i)
            
        # next phase
        if not deleted:
            eta_i = eta_i + 1





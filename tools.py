# -*- encoding=utf-8 -*-

from pathlib import Path
import shutil, os, pickle, math
import numpy as np
import matplotlib.pyplot as plt

# own
from pf_to_dem import *

#------------------------------------------------------------------------------------------------------------------------------------------ #

def create_folder(name):
    '''
    Create a new folder. If it already exists, it is erased.
    '''
    if Path(name).exists():
        shutil.rmtree(name)
    os.mkdir(name)

#------------------------------------------------------------------------------------------------------------------------------------------ #

def reduce_n_vtk_files(dict_user, dict_sample):
    '''
    Reduce the number of vtk files for phase-field and dem.

    Warning ! The pf and dem files are not synchronized...
    '''
    if dict_user['n_max_vtk_files'] != None:
        # Phase Field files

        # compute the frequency
        if dict_user['j_total']-1 > dict_user['n_max_vtk_files']:
            f_save = (dict_user['j_total']-1)/(dict_user['n_max_vtk_files']-1)
        else :
            f_save = 1
        # post proccess index
        i_save = 0

        # iterate on time 
        for iteration in range(dict_user['j_total']):
            iteration_str = index_to_str(iteration) # from pf_to_dem.py 
            if iteration >= f_save*i_save:
                i_save_str = index_to_str(i_save) # from pf_to_dem.py
                # rename .pvtu
                os.rename('vtk/pf_'+iteration_str+'.pvtu','vtk/pf_'+i_save_str+'.pvtu')
                # write .pvtu to save all vtk
                file = open('vtk/pf_'+i_save_str+'.pvtu','w')
                file.write('''<?xml version="1.0"?>
                <VTKFile type="PUnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
                \t<PUnstructuredGrid GhostLevel="1">
                \t\t<PPointData>
                \t\t\t<PDataArray type="Float64" Name="as"/>
                \t\t\t<PDataArray type="Float64" Name="kc"/>''')
                for i_grain in range(len(dict_sample['L_etai_map'])):
                    file.write('''\t\t\t<PDataArray type="Float64" Name="eta'''+str(i_grain+1)+'''"/>''')
                file.write('''\t\t\t<PDataArray type="Float64" Name="c"/>
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
                    line = line + '''\t\t<Piece Source="pf_'''+i_save_str+'''_'''+str(i_proc)+'''.vtu"/>\n'''
                file.write(line)
                file.write('''\t</PUnstructuredGrid>
                </VTKFile>''')
                file.close()
                # rename .vtk
                for i_proc in range(dict_user['n_proc']):
                    os.rename('vtk/pf_'+iteration_str+'_'+str(i_proc)+'.vtu','vtk/pf_'+i_save_str+'_'+str(i_proc)+'.vtu')
                i_save = i_save + 1 
            else:
                # delete files
                os.remove('vtk/pf_'+iteration_str+'.pvtu')
                for i_proc in range(dict_user['n_proc']):
                    os.remove('vtk/pf_'+iteration_str+'_'+str(i_proc)+'.vtu')
        # .e file
        os.remove('vtk/pf_out.e')
        # other files
        j = 0
        j_str = index_to_str(j)
        filepath = Path('vtk/pf_other_'+j_str+'.pvtu')
        while filepath.exists():
            for i_proc in range(dict_user['n_proc']):
                os.remove('vtk/pf_other_'+j_str+'_'+str(i_proc)+'.vtu')
            os.remove('vtk/pf_other_'+j_str+'.pvtu')
            j = j + 1
            j_str = index_to_str(j)
            filepath = Path('vtk/pf_other_'+j_str+'.pvtu')

#------------------------------------------------------------------------------------------------------------------------------------------ #

def save_mesh_database(dict_user, dict_sample):
    '''
    Save mesh database.
    '''
    # creating a database
    if not Path('mesh_map.database').exists():
        dict_data = {
        'n_proc': dict_user['n_proc'],
        'n_mesh_x': len(dict_sample['x_L']),
        'n_mesh_y': len(dict_sample['y_L']),
        'L_L_i_XY_used': dict_sample['L_L_i_XY_used'],
        }
        dict_database = {'Run_1': dict_data}
        with open('mesh_map.database', 'wb') as handle:
                pickle.dump(dict_database, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # updating a database
    else :
        with open('mesh_map.database', 'rb') as handle:
            dict_database = pickle.load(handle)
        dict_data = {
        'n_proc': dict_user['n_proc'],
        'n_mesh_x': len(dict_sample['x_L']),
        'n_mesh_y': len(dict_sample['y_L']),
        'L_L_i_XY_used': dict_sample['L_L_i_XY_used']
        }   
        mesh_map_known = False
        for i_run in range(1,len(dict_database.keys())+1):
            if dict_database['Run_'+str(int(i_run))]['n_proc'] == dict_data['n_proc'] and \
               dict_database['Run_'+str(int(i_run))]['n_mesh_x'] == dict_data['n_mesh_x'] and \
               dict_database['Run_'+str(int(i_run))]['n_mesh_y'] == dict_data['n_mesh_y']:
                mesh_map_known = True
        # new entry
        if not mesh_map_known: 
            key_entry = 'Run_'+str(int(len(dict_database.keys())+1))
            dict_database[key_entry] = dict_data
            with open('mesh_map.database', 'wb') as handle:
                pickle.dump(dict_database, handle, protocol=pickle.HIGHEST_PROTOCOL)

#------------------------------------------------------------------------------------------------------------------------------------------ #

def check_mesh_database(dict_user, dict_sample):
    '''
    Check mesh database.
    '''
    if Path('mesh_map.database').exists():
        with open('mesh_map.database', 'rb') as handle:
            dict_database = pickle.load(handle)
        dict_data = {
        'n_proc': dict_user['n_proc'],
        'n_mesh_x': len(dict_sample['x_L']),
        'n_mesh_y': len(dict_sample['y_L'])
        }   
        mesh_map_known = False
        for i_run in range(1,len(dict_database.keys())+1):
            if dict_database['Run_'+str(int(i_run))]['n_proc'] == dict_data['n_proc'] and \
               dict_database['Run_'+str(int(i_run))]['n_mesh_x'] == dict_data['n_mesh_x'] and \
               dict_database['Run_'+str(int(i_run))]['n_mesh_y'] == dict_data['n_mesh_y']:
                mesh_map_known = True
                i_known = i_run
        if mesh_map_known :
            dict_sample['Map_known'] = True
            dict_sample['L_L_i_XY_used'] = dict_database['Run_'+str(int(i_known))]['L_L_i_XY_used']
        else :
            dict_sample['Map_known'] = False
    else :
        dict_sample['Map_known'] = False

#------------------------------------------------------------------------------------------------------------------------------------------ #

def plot_sum_mean_etai_c(dict_user, dict_sample):
    '''
    Plot figure illustrating the sum and the mean of etai and c.
    '''
    # compute tracker
    # sum
    s_eta_i = 0
    if dict_user['L_L_sum_eta_i'] == []:
        for i_grain in range(len(dict_sample['L_etai_map'])):
            dict_user['L_L_sum_eta_i'].append([np.sum(dict_sample['L_etai_map'][i_grain])])
            s_eta_i = s_eta_i + np.sum(dict_sample['L_etai_map'][i_grain])
    else :
        for i_grain in range(len(dict_sample['L_etai_map'])):
            dict_user['L_L_sum_eta_i'][i_grain].append(np.sum(dict_sample['L_etai_map'][i_grain]))
            s_eta_i = s_eta_i + np.sum(dict_sample['L_etai_map'][i_grain])
    dict_user['L_sum_c'].append(np.sum(dict_sample['c_map']))
    dict_user['L_sum_mass'].append(1/dict_user['V_m']*s_eta_i+np.sum(dict_sample['c_map']))
    # mean
    m_eta_i = 0
    if dict_user['L_L_m_eta_i'] == []:
        for i_grain in range(len(dict_sample['L_etai_map'])):
            dict_user['L_L_m_eta_i'].append([np.mean(dict_sample['L_etai_map'][i_grain])])
            m_eta_i = m_eta_i + np.mean(dict_sample['L_etai_map'][i_grain])
    else :
        for i_grain in range(len(dict_sample['L_etai_map'])):
            dict_user['L_L_m_eta_i'][i_grain].append(np.mean(dict_sample['L_etai_map'][i_grain]))
            m_eta_i = m_eta_i + np.mean(dict_sample['L_etai_map'][i_grain])
    dict_user['L_m_c'].append(np.mean(dict_sample['c_map']))
    dict_user['L_m_mass'].append(1/dict_user['V_m']*m_eta_i+np.mean(dict_sample['c_map']))

    # plot sum eta_i, c
    if 'sum_etai_c' in dict_user['L_figures']:
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,figsize=(16,9))
        for i_grain in range(len(dict_sample['L_L_sum_eta_i'])):
            ax1.plot(dict_user['L_L_sum_eta_i'][i_grain])
        ax1.set_title(r'$\Sigma\eta_i$')
        ax3.plot(dict_user['L_sum_c'])
        ax3.set_title(r'$\Sigma C$')
        ax4.plot(dict_user['L_sum_mass'])
        ax4.set_title(r'$\Sigma mass$')
        fig.tight_layout()
        fig.savefig('plot/sum_etai_c.png')
        plt.close(fig)

    # plot mean eta_i, c
    if 'mean_etai_c' in dict_user['L_figures']:
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,figsize=(16,9))
        for i_grain in range(len(dict_user['L_L_m_eta_i'])):
            ax1.plot(dict_user['L_L_m_eta_i'][i_grain])
        ax1.set_title(r'Mean $\eta_i$')
        ax3.plot(dict_user['L_m_c'])
        ax3.set_title(r'Mean $c$')
        ax4.plot(dict_user['L_m_mass'])
        ax4.set_title(r'Mean mass')
        fig.tight_layout()
        fig.savefig('plot/mean_etai_c.png')
        plt.close(fig)

#------------------------------------------------------------------------------------------------------------------------------------------ #

def compute_mass(dict_user, dict_sample):
    '''
    Compute the mass at a certain time.
     
    Mass is sum of etai and c.
    '''
    # sum of masses
    L_sum_eta_i_tempo = []
    s_eta_i = 0
    for i_grain in range(len(dict_sample['L_etai_map'])):
        L_sum_eta_i_tempo.append([np.sum(dict_sample['L_etai_map'][i_grain])])
        s_eta_i = s_eta_i + np.sum(dict_sample['L_etai_map'][i_grain])
    dict_user['L_sum_eta_i_tempo'] = L_sum_eta_i_tempo
    dict_user['sum_c_tempo'] = np.sum(dict_sample['c_map'])
    dict_user['sum_mass_tempo'] = 1/dict_user['V_m']*s_eta_i+np.sum(dict_sample['c_map'])

#------------------------------------------------------------------------------------------------------------------------------------------ #

def compute_mass_loss(dict_user, dict_sample, tracker_key):
    '''
    Compute the mass loss from the previous compute_mass() call.
     
    Plot in the given tracker.
    Mass is sum of etai and c.
    '''
    # delta masses
    if dict_user['L_'+tracker_key+'_eta_i'] == []:
        s_eta_i = 0
        for i_grain in range(len(dict_sample['L_etai_map'])):
            detai = np.sum(dict_sample['L_etai_map'][i_grain]) - dict_user['L_sum_eta_i_tempo'][i_grain]
            s_eta_i = s_eta_i + np.sum(dict_sample['L_etai_map'][i_grain])
            # save
            dict_user['L_'+tracker_key+'_eta_i'].append([detai])
    else : 
        s_eta_i = 0
        for i_grain in range(len(dict_sample['L_etai_map'])):
            detai = np.sum(dict_sample['L_etai_map'][i_grain]) - dict_user['L_sum_eta_i_tempo'][i_grain]
            s_eta_i = s_eta_i + np.sum(dict_sample['L_etai_map'][i_grain])
            # save
            dict_user['L_'+tracker_key+'_eta_i'][i_grain].append(detai)
    dc = np.sum(dict_sample['c_map']) - dict_user['sum_c_tempo']
    dm = 1/dict_user['V_m']*s_eta_i+np.sum(dict_sample['c_map']) - dict_user['sum_mass_tempo']
    # save
    dict_user[tracker_key+'_c'].append(dc)
    dict_user[tracker_key+'_m'].append(dm)

    # plot
    if 'mass_loss' in dict_user['L_figures']:
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,figsize=(16,9))
        for i_grain in range(len(dict_sample['L_'+tracker_key+'_eta_i'])):
            ax1.plot(dict_user['L_'+tracker_key+'_eta_i'][i_grain])
        ax1.set_title(r'$\eta_i$ loss')
        ax3.plot(dict_user[tracker_key+'_c'])
        ax3.set_title(r'$c$ loss')
        ax4.plot(dict_user[tracker_key+'_m'])
        ax4.set_title(r'mass loss')
        fig.tight_layout()
        fig.savefig('plot/'+tracker_key+'.png')
        plt.close(fig)

#------------------------------------------------------------------------------------------------------------------------------------------ #

def plot_performances(dict_user, dict_sample):
    '''
    Plot figure illustrating the time performances of the algorithm.
    '''
    if 'performances' in dict_user['L_figures']:
        fig, (ax1) = plt.subplots(nrows=1,ncols=1,figsize=(16,9))
        ax1.plot(dict_user['L_t_dem'], label='DEM')
        ax1.plot(dict_user['L_t_pf'], label='PF')
        ax1.plot(dict_user['L_t_dem_to_pf'], label='DEM to PF')
        ax1.plot(dict_user['L_t_pf_to_dem_1'], label='PF to DEM 1')
        ax1.plot(dict_user['L_t_pf_to_dem_2'], label='PF to DEM 2')
        ax1.legend()
        ax1.set_title('Performances (s)')
        ax1.set_xlabel('Iterations (-)')
        fig.tight_layout()
        fig.savefig('plot/performances.png')
        plt.close(fig)
                
#------------------------------------------------------------------------------------------------------------------------------------------ #

def plot_dem(dict_user, dict_sample):
    '''
    Plot figure illustrating the overlap and force transmitted.
    '''
    # Need to be adapted to multiple contacts  

    if 'overlap' in dict_user['L_figures']:
        fig, (ax1) = plt.subplots(nrows=1,ncols=1,figsize=(16,9))
        for i_contact in range(len(dict_user['L_L_overlap'])):
            ax1.plot(dict_user['L_L_overlap'][i_contact])
        ax1.set_xlabel('iterations (-)')
        ax1.set_ylabel('overlap in DEM (-)')
        fig.tight_layout()
        fig.savefig('plot/dem_overlap.png')
        plt.close(fig)
    if 'normal_force' in dict_user['L_figures']:
        fig, (ax1) = plt.subplots(nrows=1,ncols=1,figsize=(16,9))
        for i_contact in range(len(dict_user['L_L_normal_force'])):
            ax1.plot(dict_user['L_L_normal_force'][i_contact])
        ax1.plot([0, len(dict_user['L_normal_force'])-1],\
                 [dict_user['force_applied'], dict_user['force_applied']], color='k', label='target')
        ax1.set_xlabel('iterations (-)')
        ax1.set_ylabel('normal force (-)')
        fig.tight_layout()
        fig.savefig('plot/dem_normal_force.png')
        plt.close(fig)

#------------------------------------------------------------------------------------------------------------------------------------------ #

def plot_contact(dict_user, dict_sample):
    '''
    Plot figure illustrating the contact characteristics.
    '''
    # Need to be adapted to multiple contacts
    if 'contact_volume' in dict_user['L_figures']:
        fig, (ax1) = plt.subplots(nrows=1,ncols=1,figsize=(16,9))
        for i_contact in range(len(dict_user['L_L_contact_volume'])):
            ax1.plot(dict_user['L_L_contact_volume'][i_contact])
        ax1.set_xlabel('iterations (-)')
        ax1.set_ylabel('contact volume (-)')
        fig.tight_layout()
        fig.savefig('plot/contact_volume.png')
        plt.close(fig)
    if 'contact_surface' in dict_user['L_figures']:
        fig, (ax1) = plt.subplots(nrows=1,ncols=1,figsize=(16,9))
        for i_contact in range(len(dict_user['L_L_contact_surface'])):
            ax1.plot(dict_user['L_L_contact_surface'][i_contact])
        ax1.set_xlabel('iterations (-)')
        ax1.set_ylabel('contact surface (-)')
        fig.tight_layout()
        fig.savefig('plot/contact_surface.png')
        plt.close(fig)

#------------------------------------------------------------------------------------------------------------------------------------------ #

def plot_as_pressure(dict_user, dict_sample):
    '''
    Plot figure illustrating the solid activity and pressure at the contact.
    '''
    if 'as' in dict_user['L_figures']:
        fig, (ax1) = plt.subplots(nrows=1,ncols=1,figsize=(16,9))
        for i_contact in range(len(dict_user['L_L_contact_as'])):
            ax1.plot(dict_user['L_L_contact_as'][i_contact])
        ax1.set_xlabel('iterations (-)')
        ax1.set_ylabel('solid activity (-)')
        fig.tight_layout()
        fig.savefig('plot/contact_as.png')
        plt.close(fig)

        # map
        fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
        im = ax1.imshow(dict_sample['as_map'], interpolation = 'nearest', extent=(dict_sample['x_L'][0],dict_sample['x_L'][-1],dict_sample['y_L'][0],dict_sample['y_L'][-1]))
        fig.colorbar(im, ax=ax1)
        ax1.set_title(r'Map of solid activity',fontsize = 30)
        fig.tight_layout()
        fig.savefig('plot/as_map.png')
        plt.close(fig)

    if 'pressure' in dict_user['L_figures']:
        fig, (ax1) = plt.subplots(nrows=1,ncols=1,figsize=(16,9))
        for i_contact in range(len(dict_user['L_L_contact_pressure'])):
            ax1.plot(dict_user['L_L_contact_pressure'][i_contact])
        ax1.set_xlabel('iterations (-)')
        ax1.set_ylabel('solid pressure (-)')
        fig.tight_layout()
        fig.savefig('plot/contact_pressure.png')
        plt.close(fig)

#------------------------------------------------------------------------------------------------------------------------------------------ #

def plot_displacement(dict_user, dict_sample):
    '''
    Plot figure illustrating the cumulative displacement.
    '''
    # pp data
    L_L_strain = []
    for i_grain in range(len(dict_user['L_L_displacement'][0])):
        L_strain = []
        for i_displacement in range(len(dict_user['L_L_displacement'])):
            if i_displacement == 0:
                L_displacement_cum = [dict_user['L_L_displacement'][i_displacement][i_grain][2]]
            else : 
                L_displacement_cum.append(L_displacement_cum[-1]+dict_user['L_L_displacement'][i_displacement][i_grain][2])
            L_strain.append(L_displacement_cum[-1])
        L_L_strain.append(L_strain)
    # plot
    fig, (ax1) = plt.subplots(nrows=1,ncols=1,figsize=(16,9))
    for i_grain in range(len(L_L_strain)):
        ax1.plot(L_L_strain[i_grain])
    ax1.set_xlabel('iterations (-)')
    ax1.set_ylabel('vertical strain (-)')
    fig.tight_layout()
    fig.savefig('plot/vertical_strain.png')
    plt.close(fig)

#------------------------------------------------------------------------------------------------------------------------------------------ #

def plot_config_ic(dict_user, dict_sample):
    '''
    Plot the initial configuration maps.
    '''
    if 'maps_ic' in dict_user['L_figures']:
        # solute
        fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
        im = ax1.imshow(dict_sample['c_map'], interpolation = 'nearest', extent=(dict_sample['x_L'][0],dict_sample['x_L'][-1],dict_sample['y_L'][0],dict_sample['y_L'][-1]))
        fig.colorbar(im, ax=ax1)
        ax1.set_title(r'Map of solute',fontsize = 30)
        fig.tight_layout()
        fig.savefig('plot/ic/ic_map_solute.png')
        plt.close(fig)
 
        s_eta_i = np.zeros((dict_user['n_mesh_y'], dict_user['n_mesh_x']))
        # eta_i
        for i_eta in range(len(dict_sample['L_etai_map'])):
            fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
            im = ax1.imshow(dict_sample['L_etai_map'][i_eta], interpolation = 'nearest', extent=(dict_sample['x_L'][0],dict_sample['x_L'][-1],dict_sample['y_L'][0],dict_sample['y_L'][-1]))
            fig.colorbar(im, ax=ax1)
            ax1.set_title(r'$\eta$'+str(i_eta),fontsize = 30)
            fig.tight_layout()
            fig.savefig('plot/ic/ic_map_eta'+str(i_eta)+'.png')
            plt.close(fig)

            # sum
            s_eta_i = s_eta_i + dict_sample['L_etai_map'][i_eta]
        
        # sum of etas
        fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
        im = ax1.imshow(s_eta_i, interpolation = 'nearest', extent=(dict_sample['x_L'][0],dict_sample['x_L'][-1],dict_sample['y_L'][0],dict_sample['y_L'][-1]), vmax=1)
        fig.colorbar(im, ax=ax1)
        ax1.set_title(r'$\Sigma\eta$',fontsize = 30)
        fig.tight_layout()
        fig.savefig('plot/ic/ic_s_etas.png')
        plt.close(fig)

#------------------------------------------------------------------------------------------------------------------------------------------ #

def plot_config(dict_user, dict_sample):
    '''
    Plot the current configuration maps.
    '''
    if 'configuration_c' in dict_user['L_figures']:
        # solute
        fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
        im = ax1.imshow(dict_sample['c_map'], interpolation = 'nearest', extent=(dict_sample['x_L'][0],dict_sample['x_L'][-1],dict_sample['y_L'][0],dict_sample['y_L'][-1]))
        fig.colorbar(im, ax=ax1)
        ax1.set_title(r'Map of solute',fontsize = 30)
        fig.tight_layout()
        fig.savefig('plot/configuration/c_'+str(dict_sample['i_DEMPF_ite'])+'.png')
        plt.close(fig)
 
    if 'configuration_eta' in dict_user['L_figures'] :   
        s_eta_i = np.zeros((dict_user['n_mesh_y'], dict_user['n_mesh_x']))
        # eta_i
        for i_eta in range(len(dict_sample['L_etai_map'])):
            fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
            im = ax1.imshow(dict_sample['L_etai_map'][i_eta], interpolation = 'nearest', extent=(dict_sample['x_L'][0],dict_sample['x_L'][-1],dict_sample['y_L'][0],dict_sample['y_L'][-1]))
            fig.colorbar(im, ax=ax1)
            ax1.set_title(r'$\eta$'+str(i_eta),fontsize = 30)
            fig.tight_layout()
            fig.savefig('plot/configuration/eta_'+str(i_eta)+'_'+str(dict_sample['i_DEMPF_ite'])+'.png')
            plt.close(fig)

            # sum
            s_eta_i = s_eta_i + dict_sample['L_etai_map'][i_eta]
        
        # sum of etas
        fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
        im = ax1.imshow(s_eta_i, interpolation = 'nearest', extent=(dict_sample['x_L'][0],dict_sample['x_L'][-1],dict_sample['y_L'][0],dict_sample['y_L'][-1]), vmax=1)
        fig.colorbar(im, ax=ax1)
        ax1.set_title(r'$\Sigma\eta$',fontsize = 30)
        fig.tight_layout()
        fig.savefig('plot/configuration/sum_eta_'+str(dict_sample['i_DEMPF_ite'])+'.png')
        plt.close(fig)


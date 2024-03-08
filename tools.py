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

def tuplet_to_list(tuplet):
    '''
    Convert a tuplet into lists.
    '''
    L_x = []
    L_y = []
    p_center = np.array([0,0])
    n_mean = 0
    for v in tuplet:
        L_x.append(v[0])
        L_y.append(v[1])
        p_center = p_center + np.array([v[0], v[1]])
        n_mean = n_mean + 1
    L_x.append(L_x[0])
    L_y.append(L_y[0])
    p_center = p_center/n_mean
    # translate center to the point (0,0)
    for i in range(len(L_x)):
        L_x[i] = L_x[i] - p_center[0]
        L_y[i] = L_y[i] - p_center[1]
    return L_x, L_y

#------------------------------------------------------------------------------------------------------------------------------------------ #

def tuplet_to_list_no_centerized(tuplet):
    '''
    Convert a tuplet into lists.
    '''
    L_x = []
    L_y = []
    n_mean = 0
    for v in tuplet:
        L_x.append(v[0])
        L_y.append(v[1])
        n_mean = n_mean + 1
    L_x.append(L_x[0])
    L_y.append(L_y[0])
    return L_x, L_y

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
                \t\t\t<PDataArray type="Float64" Name="kc"/>
                \t\t\t<PDataArray type="Float64" Name="eta1"/>
                \t\t\t<PDataArray type="Float64" Name="eta2"/>
                \t\t\t<PDataArray type="Float64" Name="c"/>
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

        # DEM files

        # compute the frequency
        if 2*dict_user['n_DEMPF_ite']-1 > dict_user['n_max_vtk_files']:
            f_save = (2*dict_user['n_DEMPF_ite']-1)/(dict_user['n_max_vtk_files']-1)
        else :
            f_save = 1
        # post proccess index
        i_save = 0

        # iterate on time 
        for iteration in range(2*dict_user['n_DEMPF_ite']):
            iteration_str = str(iteration) # from pf_to_dem.py 
            if iteration >= f_save*i_save:
                i_save_str = str(i_save) # from pf_to_dem.py
                os.rename('vtk/2grains_'+iteration_str+'.vtk', 'vtk/2grains_'+i_save_str+'.vtk')
                os.rename('vtk/grain1_'+iteration_str+'.vtk', 'vtk/grain1_'+i_save_str+'.vtk')
                os.rename('vtk/grain2_'+iteration_str+'.vtk', 'vtk/grain2_'+i_save_str+'.vtk')
                i_save = i_save + 1
            else :
                os.remove('vtk/2grains_'+iteration_str+'.vtk')
                os.remove('vtk/grain1_'+iteration_str+'.vtk')
                os.remove('vtk/grain2_'+iteration_str+'.vtk')          

#------------------------------------------------------------------------------------------------------------------------------------------ #

def plot_shape_evolution(dict_user, dict_sample):
    '''
    Plot figure illustrating the evolution of grain shapes.
    '''
    with open('data/vertices.data', 'rb') as handle:
        dict_save = pickle.load(handle)
    # save initial shapes
    if dict_user['L_L_vertices_init'] == None:
        dict_user['L_L_vertices_init'] = dict_save['L_L_vertices']
    #compare current shape and initial one
    if 'shape_evolution' in dict_user['L_figures']:
        # iterate on the grains
        for i_g in range(len(dict_sample['L_center'])):
            fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
            L_x, L_y = tuplet_to_list(dict_user['L_L_vertices_init'][i_g]) # from tools.py
            ax1.plot(L_x, L_y, label='Initial')
            L_x, L_y = tuplet_to_list(dict_save['L_L_vertices'][i_g]) # from tools.py
            ax1.plot(L_x, L_y, label='Current')
            ax1.legend()
            ax1.axis('equal')
            ax1.set_title('G'+str(i_g),fontsize=20)
            plt.suptitle('Shapes evolution', fontsize=20)
            fig.tight_layout()
            if dict_user['print_all_shape_evolution']:
                fig.savefig('plot/shape_evolution/g'+str(i_g)+'_ite_'+str(dict_sample['i_DEMPF_ite'])+'.png')
            else:
                fig.savefig('plot/shape_evolution/g'+str(i_g)+'.png')
            plt.close(fig)

#------------------------------------------------------------------------------------------------------------------------------------------ #

def plot_n_vertices(dict_user, dict_sample):
    '''
    Plot figure illustrating the number of vertices used in Yade.
    '''
    # load data
    with open('data/dem_to_main.data', 'rb') as handle:
        dict_save = pickle.load(handle)
    # tracker
    if dict_user['L_L_n_v_i'] == []:
        for i in range(len(dict_save['L_n_v'])):
            dict_user['L_L_n_v_i'].append([dict_save['L_n_v'][i]/2])
    else :
        for i in range(len(dict_save['L_n_v'])):
            dict_user['L_L_n_v_i'][i].append(dict_save['L_n_v'][i]/2)

    # plot
    if 'n_vertices' in dict_user['L_figures']:
        fig, (ax1) = plt.subplots(1,1,figsize=(16,9))        
        for i in range(len(dict_user['L_L_n_v_i'])):
            ax1.plot(dict_user['L_L_n_v_i'][i], label='G'+str(i))
        ax1.legend()
        ax1.set_title('N vertices per grains')
        fig.tight_layout()
        fig.savefig('plot/n_vertices.png')
        plt.close(fig)

#------------------------------------------------------------------------------------------------------------------------------------------ #

def plot_sum_mean_etai_c(dict_user, dict_sample):
    '''
    Plot figure illustrating the sum and the mean of etai and c.
    '''
    # compute tracker
    dict_user['L_sum_eta_1'].append(np.sum(dict_sample['eta_1_map']))
    dict_user['L_sum_eta_2'].append(np.sum(dict_sample['eta_2_map']))
    dict_user['L_sum_c'].append(np.sum(dict_sample['c_map']))
    dict_user['L_sum_mass'].append(np.sum(dict_sample['eta_1_map'])+np.sum(dict_sample['eta_2_map'])+np.sum(dict_sample['c_map']))
    dict_user['L_m_eta_1'].append(np.mean(dict_sample['eta_1_map']))
    dict_user['L_m_eta_2'].append(np.mean(dict_sample['eta_2_map']))
    dict_user['L_m_c'].append(np.mean(dict_sample['c_map']))
    dict_user['L_m_mass'].append(np.mean(dict_sample['eta_1_map'])+np.mean(dict_sample['eta_2_map'])+np.mean(dict_sample['c_map']))

    # plot sum eta_i, c
    if 'sum_etai_c' in dict_user['L_figures']:
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,figsize=(16,9))
        ax1.plot(dict_user['L_sum_eta_1'])
        ax1.set_title(r'$\Sigma\eta_1$')
        ax2.plot(dict_user['L_sum_eta_2'])
        ax2.set_title(r'$\Sigma\eta_2$')
        ax3.plot(dict_user['L_sum_c'])
        ax3.set_title(r'$\Sigma C$')
        ax4.plot(dict_user['L_sum_mass'])
        ax4.set_title(r'$\Sigma\eta_1 + \Sigma\eta_2 + \Sigma c$')
        fig.tight_layout()
        fig.savefig('plot/sum_etai_c.png')
        plt.close(fig)

    # plot mean eta_i, c
    if 'mean_etai_c' in dict_user['L_figures']:
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,figsize=(16,9))
        ax1.plot(dict_user['L_m_eta_1'])
        ax1.set_title(r'Mean $\eta_1$')
        ax2.plot(dict_user['L_m_eta_2'])
        ax2.set_title(r'Mean $\eta_2$')
        ax3.plot(dict_user['L_m_c'])
        ax3.set_title(r'Mean $c$')
        ax4.plot(dict_user['L_m_mass'])
        ax4.set_title(r'Mean $\eta_1$ + Mean $\eta_2$ + Mean $c$')
        fig.tight_layout()
        fig.savefig('plot/mean_etai_c.png')
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

def plot_disp_strain_andrade(dict_user, dict_sample):
    '''
    Plot figure illustrating the displacement, the strain and the fit with the Andrade law.
    '''
    # pp displacement
    L_disp_init = [0]
    L_disp = [0]
    L_strain = [0]
    for i_disp in range(len(dict_user['L_displacement'])):
        L_disp_init.append(L_disp_init[-1]+dict_user['L_displacement'][i_disp])
        if i_disp >= 1:
            L_disp.append(L_disp[-1]+dict_user['L_displacement'][i_disp])
            L_strain.append(L_strain[-1]+dict_user['L_displacement'][i_disp]/(4*dict_user['radius']))
    # compute andrade
    L_andrade = []
    L_strain_log = []
    L_t_log = []
    mean_log_k = 0
    if len(L_strain) > 1:
        for i in range(1,len(L_strain)):
            L_strain_log.append(math.log(abs(L_strain[i])))
            L_t_log.append(math.log(i+1))
            mean_log_k = mean_log_k + (L_strain_log[-1] - 1/3*L_t_log[-1])
        mean_log_k = mean_log_k/len(L_strain) # mean k in Andrade creep law
        # compute fitted Andrade creep law
        for i in range(len(L_t_log)):
            L_andrade.append(mean_log_k + 1/3*L_t_log[i])
    # plot
    if 'disp_strain_andrade' in dict_user['L_figures']:
        fig, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(16,9))
        # displacement
        ax1.plot(L_disp)
        ax1.set_title('Displacement (m)')
        # strain
        ax2.plot(L_strain)
        ax2.set_title(r'$\epsilon_y$ (-)')
        ax2.set_xlabel('Times (-)')
        # Andrade
        ax3.plot(L_t_log, L_strain_log)
        ax3.plot(L_t_log, L_andrade, color='k', linestyle='dotted')
        ax3.set_title('Andrade creep law')
        ax3.set_ylabel(r'log(|$\epsilon_y$|) (-)')
        ax3.set_xlabel('log(Times) (-)')
        # close
        fig.tight_layout()
        fig.savefig('plot/disp_strain_andrade.png')
        plt.close(fig)
    # save
    dict_user['L_disp'] = L_disp
    dict_user['L_disp_init'] = L_disp_init
    dict_user['L_strain'] = L_strain
    dict_user['L_andrade'] = L_andrade
    dict_user['mean_log_k'] = mean_log_k

#------------------------------------------------------------------------------------------------------------------------------------------ #

def plot_maps_configuration(dict_user, dict_sample):
    '''
    Plot figure illustrating the current maps of etai and c.
    '''
    # Plot
    if 'maps' in dict_user['L_figures']:
        fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(16,9))
        # eta 1
        im = ax1.imshow(dict_sample['eta_1_map'], interpolation = 'nearest', extent=(dict_sample['x_L'][0],dict_sample['x_L'][-1],dict_sample['y_L'][0],dict_sample['y_L'][-1]))
        fig.colorbar(im, ax=ax1)
        ax1.set_title(r'Map of $\eta_1$',fontsize = 30)
        # eta 2
        im = ax2.imshow(dict_sample['eta_2_map'], interpolation = 'nearest', extent=(dict_sample['x_L'][0],dict_sample['x_L'][-1],dict_sample['y_L'][0],dict_sample['y_L'][-1]))
        fig.colorbar(im, ax=ax2)
        ax2.set_title(r'Map of $\eta_2$',fontsize = 30)
        # solute
        im = ax3.imshow(dict_sample['c_map'], interpolation = 'nearest', extent=(dict_sample['x_L'][0],dict_sample['x_L'][-1],dict_sample['y_L'][0],dict_sample['y_L'][-1]))
        fig.colorbar(im, ax=ax3)
        ax3.set_title(r'Map of solute',fontsize = 30)
        # close
        fig.tight_layout()
        if dict_user['print_all_map_config']:
            fig.savefig('plot/map_etas_solute/'+str(dict_sample['i_DEMPF_ite'])+'.png')
        else:
            fig.savefig('plot/map_etas_solute.png')
        plt.close(fig)

#------------------------------------------------------------------------------------------------------------------------------------------ #

def compute_sphericities(L_vertices):
    '''
    Compute sphericity of the particle with five parameters.

    The parameters used are the area, the diameter, the circle ratio, the perimeter and the width to length ratio sphericity.
    See Zheng, J., Hryciw, R.D. (2015) Traditional soil particle sphericity, roundness and surface roughness by computational geometry, Geotechnique, Vol 65
    '''
    # adapt list
    L_vertices_x, L_vertices_y = tuplet_to_list_no_centerized(L_vertices)
    L_vertices = []
    for i_v in range(len(L_vertices_x)):
        L_vertices.append(np.array([L_vertices_x[i_v], L_vertices_y[i_v]]))

    #Find the minimum circumscribing circle
    #look for the two farthest and nearest points
    MaxDistance = 0
    for i_p in range(0,len(L_vertices)-2):
        for j_p in range(i_p+1,len(L_vertices)-1):
            Distance = np.linalg.norm(L_vertices[i_p]-L_vertices[j_p])
            if Distance > MaxDistance :
                ij_farthest = (i_p,j_p)
                MaxDistance = Distance

    #Trial circle
    center_circumscribing = (L_vertices[ij_farthest[0]]+L_vertices[ij_farthest[1]])/2
    radius_circumscribing = MaxDistance/2
    Circumscribing_Found = True
    Max_outside_distance = radius_circumscribing
    for i_p in range(len(L_vertices)-1):
        #there is a margin here because of the numerical approximation
        if np.linalg.norm(L_vertices[i_p]-center_circumscribing) > (1+0.05)*radius_circumscribing and i_p not in ij_farthest: #vertex outside the trial circle
            Circumscribing_Found = False
            if np.linalg.norm(L_vertices[i_p]-center_circumscribing) > Max_outside_distance:
                k_outside_farthest = i_p
                Max_outside_distance = np.linalg.norm(L_vertices[i_p]-center_circumscribing)
    #The trial guess does not work
    if not Circumscribing_Found:
        L_ijk_circumscribing = [ij_farthest[0],ij_farthest[1],k_outside_farthest]
        center_circumscribing, radius_circumscribing = FindCircleFromThreePoints(L_vertices[L_ijk_circumscribing[0]],L_vertices[L_ijk_circumscribing[1]],L_vertices[L_ijk_circumscribing[2]])
        Circumscribing_Found = True
        for i_p in range(len(L_vertices)-1):
            #there is 1% margin here because of the numerical approximation
            if np.linalg.norm(L_vertices[i_p]-center_circumscribing) > (1+0.05)*radius_circumscribing and i_p not in L_ijk_circumscribing: #vertex outside the circle computed
                Circumscribing_Found = False

    #look for length and width
    length = MaxDistance
    u_maxDistance = (L_vertices[ij_farthest[0]]-L_vertices[ij_farthest[1]])/np.linalg.norm(L_vertices[ij_farthest[0]]-L_vertices[ij_farthest[1]])
    v_maxDistance = np.array([u_maxDistance[1], -u_maxDistance[0]])
    MaxWidth = 0
    for i_p in range(0,len(L_vertices)-2):
        for j_p in range(i_p+1,len(L_vertices)-1):
            Distance = abs(np.dot(L_vertices[i_p]-L_vertices[j_p],v_maxDistance))
            if Distance > MaxWidth :
                ij_width = (i_p,j_p)
                MaxWidth = Distance
    width = MaxWidth

    #look for maximum inscribed circle
    #discretisation of the grain
    l_x_inscribing = np.linspace(min(L_vertices_x),max(L_vertices_x), 100)
    l_y_inscribing = np.linspace(min(L_vertices_y),max(L_vertices_y), 100)
    #creation of an Euclidean distance map to the nearest boundary vertex
    map_inscribing = np.zeros((100, 100))
    #compute the map
    for i_x in range(100):
        for i_y in range(100):
            p = np.array([l_x_inscribing[i_x], l_y_inscribing[-1-i_y]])
            #work only if the point is inside the grain
            if P_is_inside(L_vertices, p):
                #look for the nearest vertex
                MinDistance = None
                for q in L_vertices[:-1]:
                    Distance = np.linalg.norm(p-q)
                    if MinDistance == None or Distance < MinDistance:
                        MinDistance = Distance
                map_inscribing[-1-i_y, i_x] = MinDistance
            else :
                map_inscribing[-1-i_y, i_x] = 0
    #look for the peak of the map
    index_max = np.argmax(map_inscribing)
    l = index_max//100
    c = index_max%100
    radius_inscribing = map_inscribing[l, c]

    #Compute surface of the grain 
    #Sinus law
    meanPoint = np.mean(L_vertices[:-1], axis=0)
    SurfaceParticle = 0
    for i_triangle in range(len(L_vertices)-1):
        AB = np.array(L_vertices[i_triangle]-meanPoint)
        AC = np.array(L_vertices[i_triangle+1]-meanPoint)
        SurfaceParticle = SurfaceParticle + 0.5*np.linalg.norm(np.cross(AB, AC))

    #Area Sphericity
    if Circumscribing_Found :
        SurfaceCircumscribing = math.pi*radius_circumscribing**2
        AreaSphericity = SurfaceParticle / SurfaceCircumscribing
    else :
        AreaSphericity = 1

    #Diameter Sphericity
    if Circumscribing_Found :
        DiameterSameAreaParticle = 2*math.sqrt(SurfaceParticle/math.pi)
        DiameterCircumscribing = radius_circumscribing*2
        DiameterSphericity = DiameterSameAreaParticle / DiameterCircumscribing
    else :
        DiameterSphericity = 1

    #Circle Ratio Sphericity
    DiameterInscribing = radius_inscribing*2
    CircleRatioSphericity = DiameterInscribing / DiameterCircumscribing

    #Perimeter Sphericity
    PerimeterSameAreaParticle = 2*math.sqrt(SurfaceParticle*math.pi)
    PerimeterParticle = 0
    for i in range(len(L_vertices)-1):
        PerimeterParticle = PerimeterParticle + np.linalg.norm(L_vertices[i+1]-L_vertices[i])
    PerimeterSphericity = PerimeterSameAreaParticle / PerimeterParticle

    #Width to length ratio Spericity
    WidthToLengthRatioSpericity = width / length

    return AreaSphericity, DiameterSphericity, CircleRatioSphericity, PerimeterSphericity, WidthToLengthRatioSpericity

#------------------------------------------------------------------------------------------------------------------------------------------ #

def P_is_inside(L_vertices, P):
    '''
    Determine if a point P is inside of a grain

    Make a slide on constant y. Every time a border is crossed, the point switches between in and out.
    see Franklin 1994, see Alonso-Marroquin 2009
    '''
    counter = 0
    for i_p_border in range(len(L_vertices)-1):
        #consider only points if the coordinates frame the y-coordinate of the point
        if (L_vertices[i_p_border][1]-P[1])*(L_vertices[i_p_border+1][1]-P[1]) < 0 :
            x_border = L_vertices[i_p_border][0] + (L_vertices[i_p_border+1][0]-L_vertices[i_p_border][0])*(P[1]-L_vertices[i_p_border][1])/(L_vertices[i_p_border+1][1]-L_vertices[i_p_border][1])
            if x_border > P[0] :
                counter = counter + 1
    if counter % 2 == 0:
        return False
    else :
        return True
    
#------------------------------------------------------------------------------------------------------------------------------------------ #

def FindCircleFromThreePoints(P1, P2, P3):
    '''
    Compute the circumscribing circle of a triangle defined by three points.

    https://www.geeksforgeeks.org/program-find-circumcenter-triangle-2/
    '''
    # Line P1P2 is represented as ax + by = c and line P2P3 is represented as ex + fy = g
    a, b, c = lineFromPoints(P1, P2)
    e, f, g = lineFromPoints(P2, P3)

    # Converting lines P1P2 and P2P3 to perpendicular bisectors.
    #After this, L : ax + by = c and M : ex + fy = g
    a, b, c = perpendicularBisectorFromLine(P1, P2, a, b, c)
    e, f, g = perpendicularBisectorFromLine(P2, P3, e, f, g)

    # The point of intersection of L and M gives the circumcenter
    circumcenter = lineLineIntersection(a, b, c, e, f, g)

    if np.linalg.norm(circumcenter - np.array([10**9,10**9])) == 0:
        raise ValueError('The given points do not form a triangle and are collinear...')
    else :
        #compute the radius
        radius = max([np.linalg.norm(P1-circumcenter), np.linalg.norm(P2-circumcenter), np.linalg.norm(P3-circumcenter)])

    return circumcenter, radius

#------------------------------------------------------------------------------------------------------------------------------------------ #

def lineFromPoints(P, Q):
    '''
    Function to find the line given two points

    Used in FindCircleFromThreePoints().
    The equation is c = ax + by.
    https://www.geeksforgeeks.org/program-find-circumcenter-triangle-2/
    '''
    a = Q[1] - P[1]
    b = P[0] - Q[0]
    c = a * (P[0]) + b * (P[1])
    return a, b, c

#------------------------------------------------------------------------------------------------------------------------------------------ #

def lineLineIntersection(a1, b1, c1, a2, b2, c2):
    '''
    Returns the intersection point of two lines.

    Used in FindCircleFromThreePoints().
    https://www.geeksforgeeks.org/program-find-circumcenter-triangle-2/
    '''
    determinant = a1 * b2 - a2 * b1
    if (determinant == 0):
        # The lines are parallel.
        return np.array([10**9,10**9])
    else:
        x = (b2 * c1 - b1 * c2)//determinant
        y = (a1 * c2 - a2 * c1)//determinant
        return np.array([x, y])

#------------------------------------------------------------------------------------------------------------------------------------------ #

def perpendicularBisectorFromLine(P, Q, a, b, c):
    '''
    Function which converts the input line to its perpendicular bisector.

    Used in FindCircleFromThreePoints().
    The equation is c = ax + by.
    https://www.geeksforgeeks.org/program-find-circumcenter-triangle-2/
    '''
    mid_point = [(P[0] + Q[0])//2, (P[1] + Q[1])//2]
    # c = -bx + ay
    c = -b * (mid_point[0]) + a * (mid_point[1])
    temp = a
    a = -b
    b = temp
    return a, b, c

#------------------------------------------------------------------------------------------------------------------------------------------ #

def find_i_in_array(L_search, value):
    '''
    Find the index of a value inside a list.
    '''
    # compute abs difference
    L_search = list(abs(np.array(L_search)-value))
    return L_search.index(min(L_search))

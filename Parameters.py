#------------------------------------------------------------------------------------------------------------------------------------------ #
# Librairies
#------------------------------------------------------------------------------------------------------------------------------------------#

import numpy as np

#------------------------------------------------------------------------------------------------------------------------------------------ #
# Parameters
#------------------------------------------------------------------------------------------------------------------------------------------#

def get_parameters():
    '''
    Define the parameters used in the simulation.
    '''

    #---------------------------------------------------------------------#
    # PFDEM

    n_DEMPF_ite = 70 # number of PFDEM iterations
    n_proc = 5 # number of processors used
    j_total = 0 # index global of results
    save_simulation = False # indicate if the simulation is saved
    n_max_vtk_files = None # maximum number of vtk files (can be None to save all files)

    # Select Figures to plot
    # Available:
    # ic_dem, ic_config, ic_pf_dist, ic_pf_map, ic_c_map
    # maps, contact_detection, sphericities, shape_evolution, dem, n_vertices
    L_figures = ['ic_dem', 'ic_pf_dist', 'ic_pf_map',\
                 'shape_evolution', 'dem', 'n_vertices']

    # Figure (plot all or current)
    # The maps configuration
    print_all_map_config = False # else only the current one is printed 
    # The detection of the contact by a box
    print_all_contact_detection = False # else only the current one is printed
    # Compare the shape of the grains with the initial one
    print_all_shape_evolution = False # else only the current one is printed
    # The dem plot
    print_all_dem = False # else only the current one is printed

    #---------------------------------------------------------------------#
    # Grain description

    # shape of the grain
    # Sphere_no_overlap, Sphere_cfc 
    Shape = 'Sphere_no_overlap'
    # the variance of the radius
    var_radius = 0.2
    # discretization of the grain
    n_phi = 80

    # Details (parameters used in specific condition)
    # Sphere_no_overlap
    if Shape == 'Sphere_no_overlap':
        # maximum number of iteration for ic generation
        n_max_gen_ic = 100
        # the radius o
        radius = 1 # m
        # Number of grains
        n_grain = 10
    # Sphere_cfc
    elif Shape == 'Sphere_cfc':
        # the variance of the position in the cfc
        var_pos = 0.05
        # the number of grains in the line
        n_grains_x = 4
        # the number of lines
        n_lines = 7

    #---------------------------------------------------------------------#
    # Wall description 
    # Wall but what about periodic bc ?  
    
    # coordinates
    x_min_wall = 0
    if Shape == 'Sphere_no_overlap':
        x_max_wall = 10*radius + x_min_wall
    elif Shape == 'Sphere_cfc':
        x_max_wall = 2*1*n_grains_x + x_min_wall
    y_min_wall = 0
    # y_max_wall depends in the case
    if Shape == 'Sphere_no_overlap':
        # User can give value
        y_max_wall = (x_max_wall-x_min_wall)/2 + y_min_wall
    elif Shape == 'Sphere_cfc':
        # Value computed
        y_max_wall = n_lines/n_grains_x*(x_max_wall-x_min_wall) + y_min_wall

    #---------------------------------------------------------------------#
    # DEM (Yade)

    # DEM material parameters
    # Young modulus
    E = 1e8 # Pa
    # Poisson ratio
    Poisson = 0.3
    
    # steady state detection
    n_ite_max = 1000 # maximum number of iteration during a DEM step
    n_steady_state_detection = 100 # number of iterations considered in the window
    # the difference between max and min < tolerance * force_applied
    # + the force applied must be contained in this window
    steady_state_detection = 0.05

    # sollicitation on the top wall
    force_applied = 3e6 # N
    # controler coefficient
    k_control_force = E # m/N
    # limiting d_y_max
    if Shape == 'Sphere_no_overlap':
        d_y_limit = 0.005*radius
    elif Shape == 'Sphere_cfc':
        radius = (x_max_wall-x_min_wall)/(2*n_grains_x)
        d_y_limit = 0.005*radius

    # Contact box detection
    eta_contact_box_detection = 0.25 # value of the phase field searched to determine the contact box

    #---------------------------------------------------------------------#
    # Phase-Field (Moose)

    # Mesh
    margin_domain = 0.2*radius
    size_mesh_x = 0.01
    size_mesh_y = 0.01
    m_size_mesh = (size_mesh_x + size_mesh_y)/2

    # factor to distribute pf variable
    factor_pf = 1.5

    # PF material parameters
    # the energy barrier
    Energy_barrier = 1
    # number of mesh in the interface
    n_int = 6
    # the interface thickness
    w = m_size_mesh*n_int
    # the gradient coefficient
    kappa_eta = Energy_barrier*w*w/9.86
    # the mobility
    Mobility_eff = 1

    # kinetics of dissolution and precipitation
    # it affects the tilting coefficient in Ed
    k_diss = 0.1 # mol.m-2.s-1
    k_prec = 0.1

    # molar concentration at the equilibrium
    C_eq = 1 # number of C_ref, mol m-3

    # diffusion of the solute
    D_solute = 10 # m2 s-1
    n_struct_element = int(round(radius*0.10/m_size_mesh,0))
    struct_element = np.array(np.ones((n_struct_element,n_struct_element)), dtype=bool) # for dilation

    # Aitken method
    # the time stepping and duration of one PF simualtion
    # n_t_PF*dt_PF gives the total time duration
    n_t_PF = 3 # number of iterations
    # level 0
    dt_PF_0 = 0.2 # time step
    # level 1
    dt_PF_1 = dt_PF_0/2
    m_ed_contact_1 = 0.1
    # level 2
    dt_PF_2 = dt_PF_1/2
    m_ed_contact_2 = 0.2

    #---------------------------------------------------------------------#
    # trackers

    L_sample_height = []
    L_L_sum_eta_i = []
    L_sum_c = []
    L_sum_mass = []
    L_L_m_eta_i = []
    L_m_c = []
    L_m_mass = []
    L_t_pf_to_dem_1 = []
    L_t_pf_to_dem_2 = []
    L_t_dem = []
    L_t_dem_to_pf = []
    L_t_pf = []
    L_L_n_v_i = []
    L_m_ed = []
    L_m_ed_contact = []
    L_m_ed_plus_contact = []
    L_m_ed_minus_contact = []
    L_m_ed_contact_point = []
    L_dt_PF = []
    L_m_AreaSphericity = []
    L_m_DiameterSphericity = []
    L_m_CircleRatioSphericity = []
    L_m_PerimeterSphericity = []
    L_m_WidthToLengthRatioSpericity = []
    L_grain_kc_map = []
    L_L_vertices_init = None

    #---------------------------------------------------------------------#
    # dictionnary

    dict_user = {
    'n_DEMPF_ite': n_DEMPF_ite,
    'n_proc': n_proc,
    'j_total': j_total,
    'save_simulation': save_simulation,
    'n_max_vtk_files': n_max_vtk_files,
    'L_figures': L_figures,
    'print_all_map_config': print_all_map_config,
    'print_all_contact_detection': print_all_contact_detection,
    'print_all_shape_evolution': print_all_shape_evolution,
    'print_all_dem': print_all_dem,
    'n_ite_max': n_ite_max,
    'n_steady_state_detection': n_steady_state_detection,
    'steady_state_detection': steady_state_detection,
    'force_applied': force_applied,
    'k_control_force': k_control_force,
    'd_y_limit': d_y_limit,
    'E': E,
    'Poisson': Poisson,
    'eta_contact_box_detection': eta_contact_box_detection,
    'Shape': Shape,
    'var_radius': var_radius,
    'n_phi': n_phi,
    'radius': radius,
    'x_min_wall': x_min_wall,
    'x_max_wall': x_max_wall,
    'y_min_wall': y_min_wall,
    'y_max_wall': y_max_wall,
    'margin_domain': margin_domain,
    'size_mesh_x': size_mesh_x,
    'size_mesh_y': size_mesh_y,
    'm_size_mesh': m_size_mesh,
    'factor_pf': factor_pf,
    'Energy_barrier': Energy_barrier,
    'n_int': n_int,
    'w_int': w,
    'kappa_eta': kappa_eta,
    'Mobility_eff': Mobility_eff,
    'k_diss': k_diss,
    'k_prec': k_prec,
    'C_eq': C_eq,
    'D_solute': D_solute,
    'struct_element': struct_element,
    'n_t_PF': n_t_PF,
    'dt_PF_0': dt_PF_0,
    'dt_PF_1': dt_PF_1,
    'Aitken_1': m_ed_contact_1,
    'dt_PF_2': dt_PF_2,
    'Aitken_2': m_ed_contact_2,
    'L_sample_height': L_sample_height,
    'L_L_sum_eta_i': L_L_sum_eta_i,
    'L_sum_c': L_sum_c,
    'L_sum_mass': L_sum_mass,
    'L_L_m_eta_i': L_L_m_eta_i,
    'L_m_c': L_m_c,
    'L_m_mass': L_m_mass,
    'L_t_pf_to_dem_1': L_t_pf_to_dem_1,
    'L_t_pf_to_dem_2': L_t_pf_to_dem_2,
    'L_t_dem': L_t_dem,
    'L_t_dem_to_pf': L_t_dem_to_pf,
    'L_t_pf': L_t_pf,
    'L_L_n_v_i': L_L_n_v_i,
    'L_m_ed': L_m_ed,
    'L_m_ed_contact': L_m_ed_contact,
    'L_m_ed_plus_contact': L_m_ed_plus_contact,
    'L_m_ed_minus_contact': L_m_ed_minus_contact,
    'L_m_ed_contact_point': L_m_ed_contact_point,
    'L_dt_PF': L_dt_PF,
    'L_m_AreaSphericity': L_m_AreaSphericity,
    'L_m_DiameterSphericity': L_m_DiameterSphericity,
    'L_m_CircleRatioSphericity': L_m_CircleRatioSphericity,
    'L_m_PerimeterSphericity': L_m_PerimeterSphericity,
    'L_m_WidthToLengthRatioSpericity': L_m_WidthToLengthRatioSpericity,
    'L_grain_kc_map': L_grain_kc_map,
    'L_L_vertices_init': L_L_vertices_init
    }

    # add specificities
    if Shape == 'Sphere_no_overlap':
        dict_user['n_grain'] = n_grain
        dict_user['n_max_gen_ic'] = n_max_gen_ic
    elif Shape == 'Sphere_cfc':
        dict_user['var_pos'] = var_pos
        dict_user['n_grains_x'] = n_grains_x
        dict_user['n_lines'] = n_lines

    return dict_user

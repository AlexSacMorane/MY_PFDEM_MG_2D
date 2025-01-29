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
    # Norrmalization
    n_dist = 100*1e-6 # m
    n_time = 24*60*60 # s
    n_mol = 0.73*1e3 * n_dist**3 # mol

    #---------------------------------------------------------------------#
    # PFDEM

    n_DEMPF_ite = 10 # number of PFDEM iterations
    n_proc = 4 # number of processors used
    j_total = 0 # index global of results
    n_max_vtk_files = None # maximum number of vtk files (can be None to save all files)

    # Select Figures to plot
    # Available:
    # n_grain_kc_map, sum_etai_c, mean_etai_c, mass_loss, performances
    # dem, all_dem, overlap, normal_force, yade_vtk
    # contact_volume, contact_surface, as, pressure
    # displacement
    # configuration_ic, configuration_eta, configuration_c, before_after
    L_figures = ['mean_etai_c','performances',\
                 'as',\
                 'displacement',
                 'configuration_eta', 'before_after']

    #---------------------------------------------------------------------#
    # DEM (Yade)

    # steady state detection
    n_ite_max = 10000 # maximum number of iteration during a DEM step
    n_steady_state_detection = 100 # window size for the steady state detection
    steady_state_detection = 0.1 # criterion for the steady state detection

    # DEM material parameters
    # Young modulus
    E = 2e14 # (kg m-1 s-2)

    # Poisson ratio
    Poisson = 0.3

    # stiffness
    kn = E
    ks = kn*Poisson

    # force applied
    force_applied =  6e13

    #---------------------------------------------------------------------#
    # Phase-Field (Moose)

    # mesh
    check_database = True

    # PF material parameters
    # the energy barrier
    Energy_barrier = 1
    # number of mesh in the interface
    n_int = 6
    # the mobility
    Mobility_eff = 1*(100*1e-6/(24*60*60))/(n_dist/n_time) # m.s-1/(m.s-1)

    # temperature
    temperature = 623 # K 
    # molar volume
    V_m = (2.2*1e-5)/(n_dist**3/n_mol) # (m3 mol-1)/(m3 mol-1)
    # constant
    R_cst = (8.32)/(n_dist**2/(n_time**2*n_mol)) # (kg m2 s-2 mol-1 K-1)/(m2 s-2 mol-1)

    # kinetics of dissolution and precipitation
    # it affects the tilting coefficient in Ed
    k_diss = 2 # mesh dependency corected in ic.py
    k_prec = k_diss # mesh dependency corected in ic.py

    # molar concentration at the equilibrium
    C_eq = (0.73*1e3)/(n_mol/n_dist**3) # (mol m-3)/(mol m-3)

    # diffusion of the solute
    n_size_film = 5  # to be computed
    D_solute = (4e-14)/(n_dist*n_dist/n_time) # (m3 s-1)/(m2 s-1) / corected in ic.py
    n_struct_element = int(round(n_size_film,0))
    struct_element = np.array(np.ones((n_struct_element, n_struct_element)), dtype=bool) # for dilation

    # the time stepping and duration of one PF simualtion
    dt_PF = (0.01*24*60*60)/n_time # time step
    # n_t_PF*dt_PF gives the total time duration
    #n_t_PF = 200 # number of iterations
    n_t_PF = 1 # number of iterations
    
    # the criteria on residual
    crit_res = 1e-3
    
    # Contact box detection
    eta_contact_box_detection = 0.1 # value of the phase field searched to determine the contact box

    #---------------------------------------------------------------------#
    # trackers

    L_delta_y_sample = []
    L_L_overlap = []
    L_L_normal_force = []
    L_L_contact_box_x = []
    L_L_contact_box_y = []
    L_L_contact_box_z = []
    L_L_contact_volume = []
    L_L_contact_surface = []
    L_L_contact_as = []
    L_L_contact_pressure = []
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
    L_grain_kc_map = []
    L_L_loss_move_pf_eta_i = []
    L_loss_move_pf_c = []
    L_loss_move_pf_m = []
    L_L_loss_kc_eta_i = []
    L_loss_kc_c = []
    L_loss_kc_m = []
    L_L_loss_pf_eta_i = []
    L_loss_pf_c = []
    L_loss_pf_m = []

    #---------------------------------------------------------------------#
    # dictionnary

    dict_user = {
    'n_dist': n_dist,
    'n_time': n_time,
    'n_mol': n_mol,
    'n_DEMPF_ite': n_DEMPF_ite,
    'n_proc': n_proc,
    'j_total': j_total,
    'n_max_vtk_files': n_max_vtk_files,
    'L_figures': L_figures,
    'n_ite_max': n_ite_max,
    'n_steady_state_detection': n_steady_state_detection,
    'steady_state_detection': steady_state_detection,
    'E': E,
    'Poisson': Poisson,
    'kn_dem': kn,
    'ks_dem': ks,
    'force_applied': force_applied,
    'check_database': check_database,
    'Energy_barrier': Energy_barrier,
    'n_int': n_int,
    'Mobility_eff': Mobility_eff,
    'temperature': temperature,
    'V_m': V_m,
    'R_cst': R_cst,
    'k_diss': k_diss,
    'k_prec': k_prec,
    'C_eq': C_eq,
    'n_size_film': n_size_film,
    'D_solute': D_solute,
    'struct_element': struct_element,
    'dt_PF': dt_PF,
    'n_t_PF': n_t_PF,
    'crit_res': crit_res,
    'eta_contact_box_detection': eta_contact_box_detection,
    'L_delta_y_sample': L_delta_y_sample,
    'L_L_overlap': L_L_overlap,
    'L_L_normal_force': L_L_normal_force,
    'L_L_contact_box_x': L_L_contact_box_x,
    'L_L_contact_box_y': L_L_contact_box_y,
    'L_L_contact_box_z': L_L_contact_box_z,
    'L_L_contact_volume': L_L_contact_volume,
    'L_L_contact_surface': L_L_contact_surface,
    'L_L_contact_as': L_L_contact_as,
    'L_L_contact_pressure': L_L_contact_pressure,
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
    'L_grain_kc_map': L_grain_kc_map,
    'L_L_loss_move_pf_eta_i': L_L_loss_move_pf_eta_i,
    'L_loss_move_pf_c': L_loss_move_pf_c,
    'L_loss_move_pf_m': L_loss_move_pf_m,
    'L_L_loss_kc_eta_i': L_L_loss_kc_eta_i,
    'L_loss_kc_c': L_loss_kc_c,
    'L_loss_kc_m': L_loss_kc_m,
    'L_L_loss_pf_eta_i': L_L_loss_pf_eta_i,
    'L_loss_pf_c': L_loss_pf_c,
    'L_loss_pf_m': L_loss_pf_m
    }

    return dict_user

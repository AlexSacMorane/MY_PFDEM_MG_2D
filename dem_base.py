# -*- encoding=utf-8 -*-

from yade import pack, utils, plot, export
import pickle
import numpy as np

# -----------------------------------------------------------------------------#
# Functions called once
# -----------------------------------------------------------------------------#

def create_materials():
    '''
    Create materials.
    '''
    O.materials.append(FrictMat(young=E, poisson=Poisson, frictionAngle=atan(0.5), density=density, label='frictMat'))
    O.materials.append(FrictMat(young=E, poisson=Poisson, frictionAngle=0, density=density, label='frictlessMat'))

# -----------------------------------------------------------------------------#

def create_grains():
    '''
    Recreate level set from data extrapolated with phase field output.
    '''
    print("Creating level set")
    for i_grain in range(len(L_sdf_i_map)):

        # grid
        spacing = L_x_L_i[i_grain][1]-L_x_L_i[i_grain][0]
        grid = RegularGrid(
            min=(min(L_x_L_i[i_grain]), min(L_y_L_i[i_grain]), -extrude_z*spacing/2),
            nGP=(len(L_x_L_i[i_grain]), len(L_y_L_i[i_grain]), extrude_z),
            spacing=spacing
        )  

        # grid
        O.bodies.append(
            levelSetBody(grid=grid,
                        distField=L_sdf_i_map[i_grain].tolist(),
                        material=0)
                        )
        O.bodies[-1].state.blockedDOFs = 'zXYZ'
        O.bodies[-1].state.pos = L_rbm_to_apply[i_grain]
        O.bodies[-1].state.refPos = L_rbm_to_apply[i_grain]

# -----------------------------------------------------------------------------#

def create_walls():
    '''
    Recreate walls.
    '''
    # -x
    O.bodies.append(wall((L_pos_w[0], 0, 0), 0, material=1))
    # +x
    O.bodies.append(wall((L_pos_w[1], 0, 0), 0, material=1))
    # -y
    O.bodies.append(wall((0, L_pos_w[2], 0), 1, material=1))
    # +y
    O.bodies.append(wall((0, L_pos_w[3], 0), 1, material=1))

# -----------------------------------------------------------------------------#

def create_plots():
    '''
    Create plots during the DEM step.
    '''
    plot.plots = {'iteration': ('f_w_control'), 'iteration ': ('pos_w_control'), \
                  'pos_w_control': ('f_w_control'), 'iteration  ': ('unbalForce')}

# -----------------------------------------------------------------------------#

def compute_dt():
    '''
    Compute the time step used in the DEM step.
    '''
    O.dt = 0.2*SpherePWaveTimeStep(radius=5*m_size, density=density, young=E)

# -----------------------------------------------------------------------------#

def create_engines():
    '''
    Create engines.

    Overlap based on the distance

    Ip2:
        kn = given
        ks = given    

    Law2:
        Fn = kn.un
        Fs = ks.us
    '''
    O.engines = [
            VTKRecorder(recorders=["lsBodies"], fileName='./vtk/ite_PFDEM_'+str(i_DEMPF_ite)+'_', iterPeriod=0, multiblockLS=True, label='initial_export'),
            ForceResetter(),
            InsertionSortCollider([Bo1_LevelSet_Aabb(), Bo1_Wall_Aabb()], verletDist=0.00),
            InteractionLoop(
                    [Ig2_LevelSet_LevelSet_ScGeom(), Ig2_Wall_LevelSet_ScGeom()],
                    [Ip2_FrictMat_FrictMat_FrictPhys(kn=MatchMaker(algo='val', val=kn), ks=MatchMaker(algo='val', val=ks))],
                    [Law2_ScGeom_FrictPhys_CundallStrack(sphericalBodies=False)]),
            PyRunner(command='applied_force()',iterPeriod=1, label='apply_force'),
            NewtonIntegrator(damping=0.1, label='newton', gravity=(0, 0, 0)),
    		PyRunner(command='add_data()',iterPeriod=1, label='data'),
            PyRunner(command='check()',iterPeriod=1, label='checker')
    ]

    if print_vtk:
        O.engines = O.engines + [VTKRecorder(recorders=["lsBodies"], fileName='./vtk/yade/', iterPeriod=1000, multiblockLS=True, label='export')]

# -----------------------------------------------------------------------------#
# Functions called multiple times
# -----------------------------------------------------------------------------#

def applied_force():
    '''
    Apply a constant force on the control wall.
    '''
    v_plate_max = 1000*(1-0.6)/(O.dt*20000) # modify here
    kp = v_plate_max/(force_applied_target*0.1)
    Fy = O.forces.f(wall_control.id)[1]
    if Fy == 0:
        wall_control.state.vel = (0, -v_plate_max, 0)  
    else :
        dF = Fy - force_applied_target
        v_try_abs = kp*abs(dF) 
        # maximal speed is applied to top wall
        if v_try_abs < v_plate_max :
            wall_control.state.vel = (0, np.sign(dF)*v_try_abs, 0)
        else :
            wall_control.state.vel = (0, np.sign(dF)*v_plate_max, 0)

# -----------------------------------------------------------------------------#

def check():
    '''
    Try to detect a steady-state.
    A maximum number of iteration is used.
    '''
    if O.iter < max(n_ite_max*0.01, n_steady_state_detection):
        return
    window = plot.data['unbalForce'][-n_steady_state_detection:]
    if O.iter > n_ite_max or max(window)<steady_state_detection:
        if print_all_dem:
            plot.plot(noShow=True).savefig('plot/dem/'+str(i_DEMPF_ite)+'.png')
            #plot.saveDataTxt('plot/dem/'+str(i_DEMPF_ite)+'.txt', vars=('iteration', 'unbalForce', 'pos_w_control', 'f_w_control'))
        if print_dem:
            plot.plot(noShow=True).savefig('plot/dem.png')
            #plot.saveDataTxt('plot/dem.txt', vars=('iteration', 'unbalForce', 'pos_w_control', 'f_w_control'))
        O.pause() # stop DEM simulation
    
# -----------------------------------------------------------------------------#

def add_data():
    '''
    Add data to plot :
        - iteration
        - unbalanced force (mean resultant forces / mean contact force)
        - position of the control plate
        - force applied on the control plate
    '''
    plot.addData(iteration=O.iter, unbalForce=round(unbalancedForce(),3),\
                f_w_control=O.forces.f(wall_control.id)[1], pos_w_control=(wall_control.state.refPos[1]-wall_control.state.pos[1])/dim_ref)
    
# -----------------------------------------------------------------------------#
# Load data
# -----------------------------------------------------------------------------#

# other
density = 2000

# from main
with open('data/main_to_dem.data', 'rb') as handle:
    dict_save = pickle.load(handle)
Poisson = dict_save['Poisson']
E = dict_save['E']
kn = dict_save['kn']
ks = dict_save['ks']
force_applied_target = dict_save['force_applied']
n_ite_max = dict_save['n_ite_max']
i_DEMPF_ite = dict_save['i_DEMPF_ite']
L_pos_w = dict_save['L_pos_w']
n_steady_state_detection = dict_save['n_steady_state_detection']
steady_state_detection = dict_save['steady_state_detection']
print_all_dem = dict_save['print_all_dem']
print_dem = dict_save['print_dem']
print_vtk = dict_save['print_vtk']
extrude_z = dict_save['extrude_z']
m_size = dict_save['m_size']

# from phase field interpolation
with open('data/level_set.data', 'rb') as handle:
    dict_save = pickle.load(handle)
L_sdf_i_map = dict_save['L_sdf_i_map']
L_rbm_to_apply = dict_save['L_rbm_to_apply']
L_x_L_i = dict_save['L_x_L_i']
L_y_L_i = dict_save['L_y_L_i']

# -----------------------------------------------------------------------------#
# Plan simulation
# -----------------------------------------------------------------------------#

# materials
create_materials()
# create grains and walls
create_grains()
create_walls() 
# define loading
wall_control = O.bodies[-1]
dim_ref = O.bodies[-1].state.refPos[1]-O.bodies[-2].state.refPos[1] 
# Engines
create_engines()
# time step
compute_dt()
# plot
create_plots()

# -----------------------------------------------------------------------------#
# MAIN DEM
# -----------------------------------------------------------------------------#

O.run()
O.wait()

# -----------------------------------------------------------------------------#
# Output
# -----------------------------------------------------------------------------#

L_displacement = []
for i_grain in range(len(L_sdf_i_map)):
    trans = np.array(O.bodies[i_grain].state.pos - O.bodies[i_grain].state.refPos)
    rot_a = O.bodies[i_grain].state.ori.toAngleAxis()[0]
    rot_v = O.bodies[i_grain].state.ori.toAngleAxis()[1]
    L_displacement.append([trans[0], trans[1],\
                           rot_a, rot_v[0], rot_v[1], rot_v[2]])
L_contact = []
for i in O.interactions:
    # work only on grain-grain contact
    if isinstance(O.bodies[i.id1].shape, LevelSet) and isinstance(O.bodies[i.id2].shape, LevelSet):
        if i.id1 < i.id2:
            id_smaller = i.id1
            id_larger  = i.id2
        else:
            id_smaller = i.id2
            id_larger  = i.id1
        L_contact.append([id_smaller, id_larger, i.geom.penetrationDepth, np.array([i.phys.normalForce[0], i.phys.normalForce[1], i.phys.normalForce[2]])])

# Save data
dict_save = {
'L_displacement': L_displacement,
'L_contact': L_contact
}
with open('data/dem_to_main.data', 'wb') as handle:
    pickle.dump(dict_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

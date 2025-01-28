# -----------------------------------------------------------------------------#
# Imports
# -----------------------------------------------------------------------------#

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

    for i_data in range(1, n_data_base+1):
        # load data
        with open('ms/level_set_part'+str(i_data)+'.data', 'rb') as handle:
            dict_save = pickle.load(handle)
        L_sdf_i_map = dict_save['L_sdf_i_map']
        L_x_L = dict_save['L_x_L']
        L_y_L = dict_save['L_y_L']
        L_rbm = dict_save['L_rbm']

        # create grain
        for i_grain in range(len(L_sdf_i_map)):
            # grid
            spacing = L_x_L[i_grain][1]-L_x_L[i_grain][0]
            grid = RegularGrid(
                min=(min(L_x_L[i_grain]), min(L_y_L[i_grain]), -extrude_z*spacing/2),
                nGP=(len(L_x_L[i_grain]), len(L_y_L[i_grain]), extrude_z),
                spacing=spacing
            )  
            # grains
            O.bodies.append(
                levelSetBody(grid=grid,
                            distField=L_sdf_i_map[i_grain].tolist(),
                            material=0)
            )
            O.bodies[-1].state.blockedDOFs = 'zXYZ'
            O.bodies[-1].state.pos = L_rbm[i_grain]
            O.bodies[-1].state.refPos = L_rbm[i_grain]
        
# -----------------------------------------------------------------------------#

def create_walls():
    '''
    Recreate walls.
    '''
    for i_wall in range(len(L_pos_w)):
        O.bodies.append(wall((L_pos_w[i_wall][0], L_pos_w[i_wall][1], L_pos_w[i_wall][2]), L_pos_w[i_wall][3], material=1))

# -----------------------------------------------------------------------------#

def create_plots():
    '''
    Create plots during the DEM step.
    '''
    plot.plots = {'iteration': ('force_applied'), 'pos_w': ('force_applied'),\
                  'iteration ': ('unbalForce', None, 'nb_contact')}

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
            VTKRecorder(recorders=["lsBodies"], fileName='./ms/ic_ite_', iterPeriod=f_vtk, multiblockLS=True, label='vtk_export'),   
            ForceResetter(),
            InsertionSortCollider([Bo1_LevelSet_Aabb(), Bo1_Wall_Aabb()], verletDist=0.00),
            InteractionLoop(
                    [Ig2_LevelSet_LevelSet_ScGeom(), Ig2_Wall_LevelSet_ScGeom()],
                    [Ip2_FrictMat_FrictMat_FrictPhys(kn=MatchMaker(algo='val', val=kn), ks=MatchMaker(algo='val', val=ks))],
                    [Law2_ScGeom_FrictPhys_CundallStrack(sphericalBodies=False)]),
    		PyRunner(command='applied_force()', iterPeriod=1, label='force'),
            NewtonIntegrator(damping=0.1, label='newton', gravity=(0, 0, 0)),
            PyRunner(command='add_data()', iterPeriod=f_data, label='data'),
            PyRunner(command='check()', iterPeriod=1, label='checker')
    ]

# -----------------------------------------------------------------------------#
# Functions called multiple times
# -----------------------------------------------------------------------------#

def applied_force():
    '''
    Apply a constant force on the control wall.
    '''
    v_plate_max = 1000*(1-0.6)/(O.dt*max_simu) # modify here
    kp = v_plate_max/(force_applied*0.1)
    Fy = O.forces.f(wall_control.id)[1]
    if Fy == 0:
        wall_control.state.vel = (0, -v_plate_max, 0)  
    else :
        dF = Fy - force_applied
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
    if O.iter > max_simu:
        plot.plot(noShow=True).savefig('ms/ic_dem.png')
        O.pause() # stop DEM simulation
    
# -----------------------------------------------------------------------------#

def add_data():
    '''
    Add data to plot.
    '''
    plot.addData(iteration=O.iter, unbalForce=round(unbalancedForce(),3), nb_contact=avgNumInteractions(),\
                 force_applied=O.forces.f(wall_control.id)[1], pos_w=(wall_control.state.refPos[1]-wall_control.state.pos[1]))
    
# -----------------------------------------------------------------------------#
# Load data
# -----------------------------------------------------------------------------#

# other
density = 2000

# from main
with open('ms/from_main_to_ic.data', 'rb') as handle:
    dict_save = pickle.load(handle)
Poisson = dict_save['Poisson']
E = dict_save['E']
kn = dict_save['kn']
ks = dict_save['ks']
force_applied = dict_save['force_applied']

# main information
with open('ms/level_set_part0.data', 'rb') as handle:
    dict_save = pickle.load(handle)
m_size = dict_save['m_size']
L_pos_w = dict_save['L_pos_w']
n_data_base = dict_save['n_data_base']
extrude_z = dict_save['extrude_z']

# data simulation
f_data   = 200
f_vtk    = 20000
max_simu = 200000

# -----------------------------------------------------------------------------#
# Plan simulation
# -----------------------------------------------------------------------------#

# materials
create_materials()
# create grains and walls
create_grains()
create_walls() 
# define loading
wall_control = O.bodies[-3]
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

# walls
L_pos_w = [O.bodies[-6].state.pos[0], # -x
           O.bodies[-5].state.pos[0], # +x
           O.bodies[-4].state.pos[1], # -y
           O.bodies[-3].state.pos[1]] # +y         

# data
plot.saveDataTxt('ms/ic_data.txt')

# dict grains
i_grain = 0
for b in O.bodies:
    if isinstance(b.shape, LevelSet):
        print('save grain', i_grain,\
              '(size = '+str(b.shape.lsGrid.nGP[0])+'-'+str(b.shape.lsGrid.nGP[1])+')')

        # compute grid
        x_L = []
        for i in range(b.shape.lsGrid.nGP[0]):
            x_L.append(b.shape.lsGrid.gridPoint(i,0,0)[0]+b.state.pos[0])    
        y_L = []
        for i in range(b.shape.lsGrid.nGP[1]):
            y_L.append(b.shape.lsGrid.gridPoint(0,i,0)[1]+b.state.pos[1])

        # compute distance field
        ls_map = np.zeros((len(y_L), len(x_L)))
        for i in range(len(x_L)):
            for j in range(len(y_L)):
                ls_map[-1-j, i] = b.shape.distField[i][j][int(b.shape.lsGrid.nGP[2]/2)]
        # save dict
        dict_save = {
                    'x_L': x_L,
                    'y_L': y_L,
                    'ls_map': ls_map
                    }
        with open('ms/from_ic_grain_'+str(i_grain)+'.data', 'wb') as handle:
            pickle.dump(dict_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # prepare for next one
        i_grain = i_grain + 1

# load data
with open('ms/level_set_part0.data', 'rb') as handle:
    dict_save = pickle.load(handle)

# dict global
dict_save = {
'n_grains': i_grain,
'L_pos_w': L_pos_w,
'm_size': m_size,
'extrude_z': dict_save['extrude_z'],
'margins': dict_save['margins']
}
with open('ms/from_ic.data', 'wb') as handle:
    pickle.dump(dict_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

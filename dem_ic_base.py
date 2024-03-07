# -*- encoding=utf-8 -*-

from yade import pack, utils, plot, export
import polyhedra_utils
import pickle, math
import numpy as np

# ------------------------------------------------------------------------------------------------------------------------------------------ #
# Own functions

def create_plots():
    '''
    Create plots during the DEM step.
    '''
    plot.plots = {'iteration': ('sample_height'),'iteration ':('normal_force','force_applied')}

# ------------------------------------------------------------------------------------------------------------------------------------------ #

def create_materials():
    '''
    Create materials.
    '''
    O.materials.append(PolyhedraMat(young=E, poisson=Poisson, frictionAngle=math.atan(0.5), density=2000, label='Grain'))
    O.materials.append(PolyhedraMat(young=10*E, poisson=Poisson, frictionAngle=0, density=2000, label='Wall'))

# ------------------------------------------------------------------------------------------------------------------------------------------ #

def compute_vertices(center, radius, n_phi):
    '''
    Compute the vertices from grain data. 
    '''
    L_vertices = ()
    for i_phi in range(n_phi):
        phi = 2*math.pi*i_phi/n_phi
        v_i = np.array([center[0]+radius*math.cos(phi), center[1]+radius*math.sin(phi)])
        # save
        L_vertices = L_vertices + ((v_i[0], v_i[1], 0,),)
        L_vertices = L_vertices + ((v_i[0], v_i[1], 1),)
    return L_vertices

# ------------------------------------------------------------------------------------------------------------------------------------------ #

def create_polyhedral():
    '''
    Create polyhedra from center and radii.
    '''
    print("Creating polyhedra")

    for i_g in range(len(L_radius)):
        # compute vertices
        L_vertices_i = compute_vertices(L_center[i_g], L_radius[i_g], n_phi)
        # create particle
        O.bodies.append(
            polyhedra_utils.polyhedra(
                O.materials[0],
                v = L_vertices_i,
                fixed = True))
        O.bodies[-1].state.refPos = O.bodies[-1].state.pos
        O.bodies[-1].state.blockedDOFs = 'zXY'

    # initial export
    vtkExporter.exportPolyhedra()

# ------------------------------------------------------------------------------------------------------------------------------------------ #

def create_walls():
    '''
    Create walls (infinite plane).
    '''
    O.bodies.append(utils.wall(x_min_wall, axis=0, sense=1 , material=O.materials[1])) 
    O.bodies.append(utils.wall(x_max_wall, axis=0, sense=-1, material=O.materials[1])) 
    O.bodies.append(utils.wall(y_min_wall, axis=1, sense=1 , material=O.materials[1])) 
    O.bodies.append(utils.wall(y_max_wall, axis=1, sense=-1, material=O.materials[1]))    
    global id_y_max_wall
    id_y_max_wall = O.bodies[-1].id

# -----------------------------------------------------------------------------#

def create_engines():
    '''
    Create engines.

    Ip2_PolyhedraMat_PolyhedraMat_PolyhedraPhys 
    Normal: 1/kn = 1/Y1 + 1/Y2 
    Shear: 1/ks = 1/Y1v1 + 1/Y2v2
    Y is the Young modulus
    v is the Poisson ratio

    Law2_PolyhedraGeom_PolyhedraPhys_Volumetric 
    F = k N
    Force is proportionnal to the volume
    '''
    O.engines = [
            ForceResetter(),
            InsertionSortCollider([Bo1_Polyhedra_Aabb(), Bo1_Wall_Aabb()]),
            InteractionLoop(
                    [Ig2_Polyhedra_Polyhedra_PolyhedraGeom(), Ig2_Wall_Polyhedra_PolyhedraGeom()],
                    [Ip2_PolyhedraMat_PolyhedraMat_PolyhedraPhys()],
                    [Law2_PolyhedraGeom_PolyhedraPhys_Volumetric()]
            ),
    		PyRunner(command='control_top()', iterPeriod=1),
            NewtonIntegrator(damping=0.5, exactAsphericalRot=True, gravity=[0, -9.81, 0], label='newton'),
    		PyRunner(command='add_data()', iterPeriod=1),
            PyRunner(command='check()', iterPeriod=1, label='checker')
    ]

# -----------------------------------------------------------------------------#

def control_top():
    '''
    Control the top wall position to apply force.
    '''
    # compute force applied on the top wall
    global F_top_wall, y_max_wall
    F_top_wall = 0
    for inter in O.interactions.all():
        if inter.isReal: # check the contact exists
            if inter.id1 == id_y_max_wall: # Force from wall to grain
                F_top_wall = F_top_wall - inter.phys.normalForce[1]
            if inter.id2 == id_y_max_wall: # Force from grain to wall
                F_top_wall = F_top_wall + inter.phys.normalForce[1]

    # control (linear)
    d_y_max_wall = k_control_force * (F_top_wall-force_applied)
    d_y_max_wall = np.sign(d_y_max_wall)*min(abs(d_y_max_wall), d_y_limit) 
    y_max_wall = y_max_wall + d_y_max_wall 
    # move wall
    O.bodies[id_y_max_wall].state.pos = [0, y_max_wall, 0]

# -----------------------------------------------------------------------------#

def add_data():
    '''
    Add data to plot :
        - sample_height
        - normal_force applied on the top wall
    '''
    sample_height = y_max_wall-y_min_wall
    normal_force = F_top_wall
    # save
    plot.addData(iteration=O.iter, sample_height=sample_height, normal_force=normal_force, force_applied=force_applied)

# -----------------------------------------------------------------------------#

def check():
    '''
    Try to detect a steady-state.
    A maximum number of iteration is used.
    '''
    if O.iter < max(n_ite_max*0.01, n_steady_state_detection):
        return
    window = plot.data['normal_force'][-n_steady_state_detection:]
    if O.iter > n_ite_max or \
       ((max(window)-min(window))<steady_state_detection*force_applied and
        max(window)>force_applied and min(window)<force_applied):
        vtkExporter.exportPolyhedra() # final export
        if print_dem_ic:
            plot.plot(noShow=True).savefig('plot/ic_dem.png')
        O.pause() # stop DEM simulation

# -----------------------------------------------------------------------------#

def compute_dt():
    '''
    Compute the time step used in the DEM step.
    '''
    O.dt = 0.5 * polyhedra_utils.PWaveTimeStep()

# ------------------------------------------------------------------------------------------------------------------------------------------ #
# Load data

# from main
with open('data/main_to_dem_ic.data', 'rb') as handle:
    dict_save = pickle.load(handle)
E = dict_save['E']
Poisson = dict_save['Poisson']
force_applied = dict_save['force_applied']
k_control_force = dict_save['k_control_force']
d_y_limit = dict_save['d_y_limit']
L_center = dict_save['L_center']
L_radius = dict_save['L_radius']
n_phi = dict_save['n_phi']
x_min_wall = dict_save['x_min_wall']
x_max_wall = dict_save['x_max_wall']
y_min_wall = dict_save['y_min_wall']
y_max_wall = dict_save['y_max_wall']
n_ite_max = dict_save['n_ite_max']
steady_state_detection = dict_save['steady_state_detection']
n_steady_state_detection = dict_save['n_steady_state_detection']
print_dem_ic = dict_save['print_dem_ic']

# ------------------------------------------------------------------------------------------------------------------------------------------ #
# Plan simulation

# vtk exporter
vtkExporter = export.VTKExporter('vtk/ic_grains')

# Plot
create_plots()
# materials
create_materials()
# create grains
create_polyhedral()
# create walls
create_walls()
# Engines
create_engines()
# time step
compute_dt()

# ------------------------------------------------------------------------------------------------------------------------------------------ #
# DEM

O.run()
O.wait()

# ------------------------------------------------------------------------------------------------------------------------------------------ #
# Output

L_pos = []
for i_g in range(len(L_radius)):
    b = O.bodies[i_g]
    L_pos.append(np.array([float(b.state.pos[0]), float(b.state.pos[1])]))

# Save data
dict_save = {
    'L_pos': L_pos,
    'y_max_wall': y_max_wall 
}
with open('data/dem_ic_to_main.data', 'wb') as handle:
    pickle.dump(dict_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

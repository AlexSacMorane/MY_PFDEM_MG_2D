# -*- encoding=utf-8 -*-

from yade import pack, utils, plot, export
import polyhedra_utils
import pickle
import numpy as np

# -----------------------------------------------------------------------------#
# Own functions

def create_plots():
    '''
    Create plots during the DEM step.
    '''
    plot.plots = {'iteration': ('sample_height'),'iteration ':('normal_force','force_applied')}

# -----------------------------------------------------------------------------#

def create_materials():
    '''
    Create materials.
    '''
    O.materials.append(PolyhedraMat(young=E, poisson=Poisson, frictionAngle=math.atan(0.5), density=2000, label='Grain'))
    O.materials.append(PolyhedraMat(young=10*E, poisson=Poisson, frictionAngle=0, density=2000, label='Wall'))

# -----------------------------------------------------------------------------#

def create_polyhedral():
    '''
    Recreate polyhedra from data extrapolated with phase field output.
    '''
    print("Creating polyhedra")

    for i_g in range(len(L_L_vertices)):
        # compute vertices
        L_vertices_i = L_L_vertices[i_g]
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
    		PyRunner(command='control_top()',iterPeriod=1),
            NewtonIntegrator(damping=0.5, exactAsphericalRot=True, gravity=[0, 0, 0], label='newton'),
    		PyRunner(command='add_data()',iterPeriod=1),
            PyRunner(command='check()',iterPeriod=1, label='checker')
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
        - iteration
        - overlap between the two particles (>0 if overlap)
    '''
    sample_height = y_max_wall-y_min_wall
    normal_force = F_top_wall
    # save
    plot.addData(iteration=O.iter, sample_height=sample_height, normal_force=normal_force, force_applied=force_applied)

# -----------------------------------------------------------------------------#

def check():
    '''
    Try to detect a wteady-state of the overlap between the two particles.
    A maximum number of iteration is used.
    '''
    if O.iter < max(n_ite_max*0.01, n_steady_state_detection):
        return
    window = plot.data['normal_force'][-n_steady_state_detection:]
    if O.iter > n_ite_max or \
       ((max(window)-min(window))<steady_state_detection*force_applied and
        max(window)>force_applied and min(window)<force_applied):
        vtkExporter.exportPolyhedra() # final export
        if print_dem:
            if print_all_dem:
                plot.plot(noShow=True).savefig('plot/dem/'+str(i_DEMPF_ite)+'.png')
            else:
                plot.plot(noShow=True).savefig('plot/dem.png')
        O.pause() # stop DEM simulation

# -----------------------------------------------------------------------------#

def compute_dt():
    '''
    Compute the time step used in the DEM step.
    '''
    O.dt = 0.1 * polyhedra_utils.PWaveTimeStep()

# ------------------------------------------------------------------------------------------------------------------------------------------ #
# Load data

# from main
with open('data/main_to_dem.data', 'rb') as handle:
    dict_save = pickle.load(handle)
E = dict_save['E']
Poisson = dict_save['Poisson']
force_applied = dict_save['force_applied']
k_control_force = dict_save['k_control_force']
d_y_limit = dict_save['d_y_limit']
x_min_wall = dict_save['x_min_wall']
x_max_wall = dict_save['x_max_wall']
y_min_wall = dict_save['y_min_wall']
y_max_wall = dict_save['y_max_wall']
n_ite_max = dict_save['n_ite_max']
steady_state_detection = dict_save['steady_state_detection']
n_steady_state_detection = dict_save['n_steady_state_detection']
print_all_dem = dict_save['print_all_dem']
print_dem = dict_save['print_dem']
i_DEMPF_ite = dict_save['i_DEMPF_ite']

# from plane interpolation
with open('data/vertices.data', 'rb') as handle:
    dict_save = pickle.load(handle)
L_L_vertices = dict_save['L_L_vertices']

# ------------------------------------------------------------------------------------------------------------------------------------------ #
# Plan simulation

# vtk exporter
vtkExporter = export.VTKExporter('vtk/grains')

# Plot
create_plots() # from dem.py
# materials
create_materials() # from dem.py
# create grains
create_polyhedral() # from dem.py
# create walls
create_walls()
# Engines
create_engines() # from dem.py
# time step
compute_dt() # from dem.py

# ------------------------------------------------------------------------------------------------------------------------------------------ #
# DEM

O.run()
O.wait()

# ------------------------------------------------------------------------------------------------------------------------------------------ #
# Output

# positions
L_pos = []
# box
L_box = []
# numbers of vertices
L_n_v = []
# displacements (translations + rotations)
L_disp_trans = []
L_disp_rot = []
# iterate on grains
for i_g in range(len(L_L_vertices)):
    # position
    L_pos.append(np.array([float(O.bodies[i_g].state.pos[0]), float(O.bodies[i_g].state.pos[1])]))
    # box
    box_dim = None
    for v in bodies[i_g].shape.v:
        dim_tempo = (v-Vector3(0,0,v[2])).norm()
        if box_dim == None:
            box_dim = dim_tempo
        else :
            if box_dim < dim_tempo:
                box_dim = dim_tempo
    L_box.append(box_dim)
    # number of vertices
    L_n_v.append(len(O.bodies[i_g].shape.v))
    # displacement
    L_disp_trans.append(np.array(O.bodies[i_g].state.pos - O.bodies[i_g].state.refPos)) 
    L_disp_rot.append(np.array(O.bodies[i_g].state.rot()))
# forces
L_normal_force = []
L_shear_force = []
# contact ids
L_contact_ids = []
# Boolean for wall contact
L_contact_wall = [] 
# contact points
L_contact_point = []
# iterate on interactions
for inter in O.interactions.all():
    if inter.isReal: # check the contact exists
        # forces
        L_normal_force.append(np.array(inter.phys.normalForce))
        L_shear_force.append(np.array(inter.phys.shearForce))
        # contact ids
        L_contact_ids.append([int(inter.id1), int(inter.id2)])
        # Boolean for wall contact
        L_contact_wall.append(isinstance(O.bodies[inter.id1].shape, Wall) or isinstance(O.bodies[inter.id2].shape, Wall))
        # contact points
        L_contact_point.append(np.array(inter.geom.contactPoint))

# Save data
dict_save = {
'L_pos': L_pos,
'L_box': L_box,
'L_n_v': L_n_v,
'L_disp_trans': L_disp_trans,
'L_disp_rot': L_disp_rot,
'L_normal_force': L_normal_force,
'L_shear_force': L_shear_force,
'L_contact_ids': L_contact_ids,
'L_contact_wall': L_contact_wall,
'L_contact_point': L_contact_point
}
with open('data/dem_to_main.data', 'wb') as handle:
    pickle.dump(dict_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

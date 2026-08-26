
import h5py
import numpy as np
import os
import time

class jorek:
  def __init__(self,key):
    self.nodes_xx  = []
    self.vertices  = []
    self.elements_size   = []
    self.traces    = {}
    self.model     = None
    self.variables = []
    self.key = key
    self.parent_key = None
    self.text = None

  def visible(self):
    print("jorek object: visible not yet implemented")

  def read_hdf5(self,file_name):

    print('read_hdf5 self :',self)

    t_wall = time.time()
    t_cpu  = time.process_time()

    self.text = os.path.basename(file_name)
    self.hdf5 = h5py.File(file_name,'r')

    if len(self.hdf5['x'][:].shape) == 3:
      self.nodes_xx   = self.hdf5['x'][:]
    else:
      self.nodes_xx   = self.hdf5['x'][:,:,0,:]
    
    self.boundary      = self.hdf5['boundary'][:]
    self.values        = self.hdf5['values'][:]   # values(node_list%n_nodes,n_tor,n_order+1,n_var)
    self.vertices      = self.hdf5['vertex'][:] - 1
    self.elements_size = self.hdf5['size'][:]

    self.model = self.hdf5['jorek_model'][0]

    print("number of nodes    : ",self.nodes_xx.shape)
    print("number of elements (vertices) : ",self.vertices.shape)
    print("number of elements (sizes)    : ",self.elements_size.shape)

    if (self.model == 303 or self.model == 307 or self.model == 600):
      self.variables = ["flux","potential","current","vorticity","density","temperature","v_par"]
    elif (self.model == 400):
      self.variables = ["flux","potential","current","vorticity","density","T_i","v_par","T_e"]
    elif (self.model == 500):
      self.variables = ["flux","potential","current","vorticity","density","temperature","v_par","neutrals"]
    elif (self.model == 502):
      self.variables = ["flux","potential","current","vorticity","density","T_i","v_par","neutrals","T_e"]
    elif (self.model == 199):
      self.variables = ["flux","potential","current","vorticity","density","temperature"]
    elif (self.model == 710):
      self.variables = ["A_3","A_R","A_Z","U_R","U_Z","U_phi","density","temperature"]
    elif (self.model == 100):
      self.variables = ["flux","potential","current","vorticity"]
    elif (self.model == -1):
      self.variables = ["density"]
    else:
      print("model not supported yet :",self.model)

    self.t_now = self.hdf5['t_now'][0]
    self.n_var = self.hdf5['n_var'][0]
    self.n_tor = self.hdf5['n_tor'][0]
    self.n_period = self.hdf5['n_period'][0]

    print(' n_tor, n_period : ',self.n_tor.item(),self.n_period.item())

    if "eta" in self.hdf5:
      print(' eta       : ',self.hdf5['eta'][0])
      print(' visco     : ',self.hdf5['visco'][0])
      print(' visco_par : ',self.hdf5['visco_par'][0])
    if "tstep" in self.hdf5:
      print(' tstep     : ',self.hdf5['tstep'][0])

    self.harmonics = list(range((int(self.n_tor.item())+1)//2))
    self.harmonics = [int(self.n_period.item()) * n for n in self.harmonics]  

    print(' harmonics : ',self.harmonics)

    if "xtime" in self.hdf5:
      self.traces['xtime']          = self.hdf5['xtime'][:]
      self.traces['energies']       = self.hdf5['energies'][:]
      self.traces['pressure_in_t']  = self.hdf5['pressure_in_t'][:]
      self.traces['pressure_out_t'] = self.hdf5['pressure_out_t'][:]
      self.traces['density_in_t']   = self.hdf5['density_in_t'][:]
      self.traces['density_out_t']  = self.hdf5['density_out_t'][:]
      self.traces['R_axis_t']       = self.hdf5['R_axis_t'][:]
      self.traces['Z_axis_t']       = self.hdf5['Z_axis_t'][:]
      self.traces['psi_axis_t']     = self.hdf5['psi_axis_t'][:]
#      for i in range(len(self.traces['xtime'])-1):
#        print(i+1,self.traces['xtime'][i+1]-self.traces['xtime'][i])
    else:
      self.traces['xtime'] = [0]

    t_wall = time.time() - t_wall
    t_cpu  = time.process_time() -t_cpu
    print('h5py timing : ',t_wall,t_cpu)

  def read_grid_hdf5(self, file_name):
    """Read only the geometry datasets required by the grid editor."""
    required = {"x", "boundary", "vertex", "size"}
    with h5py.File(file_name, "r") as hdf5:
      missing = required.difference(hdf5.keys())
      if missing:
        raise ValueError(
          "Grid file is missing required datasets: "
          + ", ".join(sorted(missing))
        )
      x = hdf5["x"][:]
      if x.ndim == 3:
        self.nodes_xx = x
      elif x.ndim == 4:
        self.nodes_xx = x[:, :, 0, :]
      else:
        raise ValueError("Grid dataset x must have three or four dimensions")
      self.boundary = hdf5["boundary"][:]
      self.vertices = hdf5["vertex"][:] - 1
      self.elements_size = hdf5["size"][:]
      self.grid_dataset_names = set(hdf5.keys())
      self.grid_only_source = self.grid_dataset_names == required
    self.text = os.path.basename(file_name)
    self.validate_grid_arrays(
      self.nodes_xx, self.boundary, self.vertices, self.elements_size
    )
    return self

  @staticmethod
  def validate_grid_arrays(nodes_xx, boundary, vertices, element_sizes):
    nodes_xx = np.asarray(nodes_xx)
    boundary = np.asarray(boundary)
    vertices = np.asarray(vertices)
    element_sizes = np.asarray(element_sizes)
    if nodes_xx.ndim != 3 or nodes_xx.shape[:2] != (2, 4):
      raise ValueError("Grid x must have shape (2, 4, n_nodes)")
    node_count = nodes_xx.shape[2]
    if boundary.ndim != 1 or len(boundary) != node_count:
      raise ValueError("Grid boundary must contain one value per node")
    if vertices.ndim != 2 or vertices.shape[0] != 4:
      raise ValueError("Grid vertex must have shape (4, n_elements)")
    element_count = vertices.shape[1]
    if element_sizes.shape != (4, 4, element_count):
      raise ValueError("Grid size must have shape (4, 4, n_elements)")
    if vertices.size and (
      np.min(vertices) < 0 or np.max(vertices) >= node_count
    ):
      raise ValueError("Grid vertex contains an out-of-range node index")

  def write_grid_hdf5(
    self, file_name, nodes_xx=None, boundary=None, vertices=None,
    element_sizes=None,
  ):
    """Write a geometry-only editor grid; no simulation datasets are added."""
    nodes_xx = self.nodes_xx if nodes_xx is None else np.asarray(nodes_xx)
    boundary = self.boundary if boundary is None else np.asarray(boundary)
    vertices = self.vertices if vertices is None else np.asarray(vertices)
    element_sizes = (
      self.elements_size
      if element_sizes is None else np.asarray(element_sizes)
    )
    self.validate_grid_arrays(
      nodes_xx, boundary, vertices, element_sizes
    )
    with h5py.File(file_name, "w") as hdf5:
      hdf5.create_dataset("x", data=nodes_xx)
      hdf5.create_dataset("boundary", data=boundary)
      hdf5.create_dataset("vertex", data=vertices + 1)
      hdf5.create_dataset("size", data=element_sizes)

  def print_keys(self):
    for key in self.hdf5.keys():
      print(key)


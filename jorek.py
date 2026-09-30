
import h5py
import hashlib
import numpy as np
import os
import shutil
import tempfile
import time


JOREK_RESTART_REQUIRED_DATASETS = frozenset({
  "x", "boundary", "vertex", "size", "values", "deltas",
  "jorek_model", "n_var", "n_tor", "n_period", "n_nodes", "n_elements",
})


def jorek_restart_hdf5_status(file_name):
  """Return ``(is_restart, reason)`` for a conservative JOREK check."""
  try:
    with h5py.File(file_name, "r") as hdf5:
      missing = JOREK_RESTART_REQUIRED_DATASETS.difference(hdf5.keys())
      if missing:
        return False, (
          "The file contains grid geometry but is not a full JOREK restart; "
          "missing required restart datasets: " + ", ".join(sorted(missing))
        )
      non_datasets = sorted(
        name for name in JOREK_RESTART_REQUIRED_DATASETS
        if not isinstance(hdf5[name], h5py.Dataset)
      )
      if non_datasets:
        return False, (
          "Required JOREK restart objects are not datasets: "
          + ", ".join(non_datasets)
        )
  except (OSError, ValueError) as error:
    return False, "Could not read the source as HDF5: {}".format(error)
  return True, ""


def is_jorek_restart_hdf5(file_name):
  """Return whether *file_name* has the mandatory JOREK restart markers."""
  return jorek_restart_hdf5_status(file_name)[0]


def validate_jorek_restart_hdf5(file_name):
  """Raise ``ValueError`` unless *file_name* is a readable JOREK restart."""
  is_restart, reason = jorek_restart_hdf5_status(file_name)
  if not is_restart:
    raise ValueError(reason)


def file_sha256(file_name, chunk_size=1024 * 1024):
  """Return the SHA-256 digest of a file without loading it into memory."""
  digest = hashlib.sha256()
  with open(file_name, "rb") as source:
    for chunk in iter(lambda: source.read(chunk_size), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _same_file_path(first, second):
  first_path = os.path.normcase(os.path.realpath(os.path.abspath(first)))
  second_path = os.path.normcase(os.path.realpath(os.path.abspath(second)))
  if first_path == second_path:
    return True
  if os.path.exists(first) and os.path.exists(second):
    try:
      return os.path.samefile(first, second)
    except OSError:
      pass
  return False


def clone_jorek_restart_hdf5(source, destination):
  """Atomically create an exact, validated copy of a JOREK restart."""
  source = os.path.abspath(source)
  destination = os.path.abspath(destination)
  if _same_file_path(source, destination):
    raise ValueError("The exported restart must not overwrite its source file")
  validate_jorek_restart_hdf5(source)

  destination_directory = os.path.dirname(destination) or os.curdir
  os.makedirs(destination_directory, exist_ok=True)
  descriptor, temporary_filename = tempfile.mkstemp(
    prefix=".jorek_restart_", suffix=".h5", dir=destination_directory,
  )
  os.close(descriptor)
  try:
    shutil.copy2(source, temporary_filename)
    validate_jorek_restart_hdf5(temporary_filename)
    source_digest = file_sha256(source)
    copied_digest = file_sha256(temporary_filename)
    if source_digest != copied_digest:
      raise OSError("Restart copy failed byte-for-byte SHA-256 validation")
    os.replace(temporary_filename, destination)
  finally:
    if os.path.exists(temporary_filename):
      os.remove(temporary_filename)
  return source_digest


def _copied_hdf5_attributes(hdf5_object):
  attributes = {}
  for name, value in hdf5_object.attrs.items():
    attributes[name] = np.array(value, copy=True) if isinstance(
      value, np.ndarray
    ) else value
  return attributes


def hdf5_inventory(file_name):
  """Return recursive HDF5 object metadata without reading dataset values."""
  inventory = {}
  with h5py.File(file_name, "r") as hdf5:
    inventory["/"] = {
      "type": "group",
      "attributes": _copied_hdf5_attributes(hdf5),
    }

    def record(name, hdf5_object):
      path = "/" + name
      entry = {
        "type": "dataset" if isinstance(hdf5_object, h5py.Dataset)
        else "group",
        "attributes": _copied_hdf5_attributes(hdf5_object),
      }
      if isinstance(hdf5_object, h5py.Dataset):
        entry.update({
          "shape": hdf5_object.shape,
          "dtype": hdf5_object.dtype.str,
          "chunks": hdf5_object.chunks,
          "compression": hdf5_object.compression,
          "compression_opts": hdf5_object.compression_opts,
        })
      inventory[path] = entry

    hdf5.visititems(record)
  return inventory

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


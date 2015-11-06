"""
OpenPMD data structures



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
# Copyright (c) 2015, Daniel Grassinger (HZDR)
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

from yt.data_objects.grid_patch import \
    AMRGridPatch
from yt.geometry.grid_geometry_handler import \
    GridIndex
from yt.data_objects.static_output import \
    Dataset
from .fields import OpenPMDFieldInfo

from yt.utilities.file_handler import \
    HDF5FileHandler

import h5py
import numpy as np
import os

# Diese Klasse legt die Eigenschaften eines Grids fest
# Momentan gibt nur ein Grid, das die ganze Simulationsbox belegt
class OpenPMDGrid(AMRGridPatch):
    _id_offset = 0
    __slots__ = ["_level_id"]
    def __init__(self, id, index, level = -1):
        AMRGridPatch.__init__(self, id, filename = index.index_filename,
                              index = index)
	# Da es nur ein Grid gibt, sind keine Kinder- oder Elterngrids vorhanden
        self.Parent = None
        self.Children = []
        self.Level = level

    def __repr__(self):
        return "OpenPMDGrid_%04i (%s)" % (self.id, self.ActiveDimensions)
# legt welche Felder und Partikel angelegt und von der Festplatte gelesen werden
# Ausserdem werden die Eigenschaften der Grids festgelegt
class OpenPMDHierarchy(GridIndex):
    grid = OpenPMDGrid

    def __init__(self, ds, dataset_type='openPMD'):
        self.dataset_type = dataset_type
        # for now, the index file is the dataset!
	self.dataset = ds
        self.index_filename = ds.parameter_filename
        self.directory = os.path.dirname(self.index_filename)
        GridIndex.__init__(self, ds, dataset_type)

    # Prueft welche Felder und Partikelfelder in der Datein enthalten sind und legt
    # somit fest, wie die Felder spaeter in yt heissen
    def _detect_output_fields(self):
        # This needs to set a self.field_list that contains all the available,
        # on-disk fields.
        # NOTE: Each should be a tuple, where the first element is the on-disk
        # fluid type or particle type.  Convention suggests that the on-disk
        # fluid type is usually the dataset_type and the on-disk particle type
        # (for a single population of particles) is "io".
	# look for fluid fields
	basePath = self.dataset._handle.attrs["basePath"]
	meshesPath = self.dataset._handle.attrs["meshesPath"]
	particlesPath = self.dataset._handle.attrs["particlesPath"]
        output_fields = []

	#Liest die Namen der allgemeinen Felder und fuegt sie output_fields hinzu
	for group in self.dataset._handle[basePath+meshesPath].keys():
	    try:
		for direction in self.dataset._handle[basePath+meshesPath+group].keys():
		    output_fields.append(group+"_"+direction)
	    except:
		output_fields.append(group)

	# fuegt die Namen zusammen mit dem Frontend der field_list hinzu
        self.field_list = [("openPMD", str(c)) for c in output_fields]

        # look for particle fields
        particle_fields = []
        
	# Hier werden die Partikelarten und Namen der Partikelfelder aus der Datei gelesen
	for particle_type in self.dataset._handle[basePath+particlesPath].keys():
	    for group in self.dataset._handle[basePath+particlesPath+particle_type].keys():
		
		try:
		    
		    key = self.dataset._handle[basePath+particlesPath+particle_type+"/"+group].keys()
		    if key == []:
			particle_fields.append(particle_type+"_"+group)
			pass			
		    else:
			for direction in key:	
			    particle_fields.append(particle_type+"_"+group+"_"+direction)
			    pass
			    
			    
		except:
		    
		    particle_fields.append(particle_type+"_"+group)
	# Die Namen der Partikelfelder werden zusammen mit der Partikelart der field_list angefuegt
	self.field_list.extend([(str(c).split("_")[0], str(c).replace(str(c).split("_")[0], "particle")) for c in particle_fields])
	        

    def _count_grids(self):
        # This needs to set self.num_grids
	# Momentan gibt es nur ein groesses Grid
	self.num_grids = 1
        

    # Die Dimensionen und die Groesse des Grid wird festgelegt
    # Da es momentan nur ein Grid gibt werden hier die Daten der Simulationsbox verwendet
    def _parse_index(self):
        # This needs to fill the following arrays, where N is self.num_grids:
	basePath = self.dataset._handle.attrs["basePath"]
	meshesPath = self.dataset._handle.attrs["meshesPath"]
	particlesPath = self.dataset._handle.attrs["particlesPath"]

        self.grid_left_edge[0] = self.dataset.domain_left_edge #(N, 3) <= float64
        self.grid_right_edge[0] = self.dataset.domain_right_edge #(N, 3) <= float64
        self.grid_dimensions[0] = self.dataset.domain_dimensions #(N, 3) <= int
        self.grid_particle_count[0] = self.dataset._handle[basePath+particlesPath+"/electrons/position/x"].shape[0]   #(N, 1) <= int
        #self.grid_levels = 1           #(N, 1) <= int
        #self.grids = np.empty(1, dtype='object') #(N, 1) <= grid objects
	
	self.grid_levels.flat[:] = 0
        self.grids = np.empty(self.num_grids, dtype='object')
	# Grids muessen initialisiert werden
        for i in range(self.num_grids):
            self.grids[i] = self.grid(i, self, self.grid_levels[i,0])
        
	
    # Diese Funktion initialisiert die Grids
    def _populate_grid_objects(self):
        # For each grid, this must call:
        #grid._prepare_grid()
        #grid._setup_dx()
        # This must also set:
        #   grid.Children <= list of child grids
        #   grid.Parent   <= parent grid
        # This is handled by the frontend because often the children must be
        # identified.
	
	#self._reconstruct_parent_child()
	
	for i in range(self.num_grids):
            self.grids[i]._prepare_grid()
            self.grids[i]._setup_dx()
        self.max_level = 0

# Ein Datenset Oabjekt einthaelt alle Informationen einer Simulation und wird
# beim yt.load() Befehl zurueck gegeben
class OpenPMDDataset(Dataset):
    _index_class = OpenPMDHierarchy
    _field_info_class = OpenPMDFieldInfo

    def __init__(self, filename, dataset_type='openPMD',
                 storage_filename=None,
                 units_override=None):
	# Hier wird festgelegt, welche Feldtypen keine Partikelfelder sind
        self.fluid_types += ('openPMD',)
	# Hier wird festgelegt, welche 
	self.particle_types = ["electrons","ions","all"]
	self.particle_types = tuple(self.particle_types)
        self.particle_types_raw = self.particle_types
	
	# Hier wird die HDF5-Datei geladen und steht an mehreren Stellen im 
	# Programm als _handle zur verfuegung 
	# Alle _handle beziehen sich auf diese Datei
	self._handle = HDF5FileHandler(filename)
        Dataset.__init__(self, filename, dataset_type,
                         units_override=units_override)
        self.storage_filename = storage_filename

    # Hier wird festgelegt in welchem Einheitensystem die Daten auf der Festplatte haben
    def _set_code_unit_attributes(self):
        # This is where quantities are created that represent the various
        # on-disk units.  These are the currently available quantities which
        # should be set, along with examples of how to set them to standard
        # values.
        #
         self.length_unit = self.quan(1.0, "m")
         self.mass_unit = self.quan(1.0, "kg")
         self.time_unit = self.quan(1.0, "s")
        
        #
        # These can also be set:
         self.velocity_unit = self.quan(1.0, "m/s")
         self.magnetic_unit = self.quan(1.0, "T")
         
    
    # Hier werden die Simulationsparameter geladen
    def _parse_parameter_file(self):
        # This needs to set up the following items.  Note that these are all
        # assumed to be in code units; domain_left_edge and domain_right_edge
        # will be updated to be in code units at a later time.  This includes
        # the cosmological parameters.

        
	#read parameters out .h5 file
	f = self._handle

	basePath = f.attrs["basePath"]
	meshesPath = f.attrs["meshesPath"]
	particlesPath = f.attrs["particlesPath"]
	positionPath = basePath+particlesPath +"/electrons/position/"

	# Hier wird die Groesse der Simulationsbox festgelegt
	#!!! Hier ist die Groesse der Simulationsbox noch hardgecodet
        self.unique_identifier = 0 # no identifier
        self.parameters  = 0 # no additional parameters  <= full of code-specific items of use
        self.domain_left_edge = np.array([ -1.49645302e-05,  -1.00407931e-06,  -3.48883032e-06]) #  <= array of float       
	self.domain_right_edge  = np.array([  1.49645302e-05,   1.41565351e-06,  1.54200006e-05]) #   <= array of float64
        self.dimensionality = 3 # <= int
	
	fshape = []	
	for i in range(3):
	    try:
		fshape.append(f[basePath+meshesPath+"/B/x"].shape[i])
	    except:
		fshape.append(1)
        self.domain_dimensions = np.array([fshape[0], fshape[2],fshape[1]], dtype="int64") #    <= array of int64	
	
	self.periodicity = (False,False,False) # <= three-element tuple of booleans
        self.current_time = f[f.attrs["basePath"]].attrs["time"] # <= simulation time in code units
	self.refine_by = 2

	
        #
        # We also set up cosmological information.  Set these to zero if
        # non-cosmological.
        
        # Es handelt sich um keine kosmologische Simulation
        self.cosmological_simulation = 0   #<= int, 0 or 1
        self.current_redshift        = 0   #<= float
        self.omega_lambda            = 0   #<= float
        self.omega_matter            = 0   #<= float
	self.hubble_constant         = 0   #<= float
	
           
    # Diese Funktion prueft ob eine mit yt.load() geladene Datei mit diesem Frontend geoeffnet werden kann
    @classmethod
    def _is_valid(self, *args, **kwargs):
	#check if openPMD standard is used        
	#return True
		
	# Ueberprueft ob die HDF5-Datei dem OpenPMD Standard entspricht
	need_attributes = ['openPMD', 'openPMDextension']
        
        valid = True
        try:
            fileh = h5py.File(args[0], mode='r')
            for na in need_attributes:
                if na not in fileh["/"].attrs.keys():
                    valid = False                    
            fileh.close()
        except:
            valid = False
        # wenn True zurueck gegeben wird entspricht die Datei dem OpenPMD Stadard
        return valid
	


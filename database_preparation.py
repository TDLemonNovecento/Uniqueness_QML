'''in this file some more or less efficient methods will be written to retrieve xyz data from files, store it in one file to make programs more efficient, and to open such a file with a lot of xyz data and reread it.
'''
from jax.config import config
config.update("jax_enable_x64", True) #increase precision from float32 to float64
import jax_representation as jrep
import numpy as np
import pickle
import copy
import os
import jax.numpy as jnp
import jax_basis as jbas

import qml
from qml.kernels import gaussian_kernel, laplacian_kernel
from qml.math import cho_solve


class compound():
    '''
    stores information of molecules
    how can I force Z entries to be of type float?
    '''
    def __init__(self, filename, Z, R, N):
        self.Z = Z
        self.R = R
        self.N = N
        self.filename = filename
        self.energy = None

    def add_energy(self, energy):
        self.energy = energy
    
    def heavy_atoms(self):
        unique, counts = jnp.unique(self.Z, return_counts = True)
        dictionary =dict(zip(unique,counts))
        try:
            heavy_atoms = self.Z.size - dictionary[1]
        except KeyError:
            print("In file %s no hydrogens were reported" % self.filename)
            heavy_atoms = self.Z.size

        return(heavy_atoms)

class derivative_results():
    '''
    stores dZ, dR, ect. data
    '''
    #reduce storage:
    __slots__ = ['filename', 'Z', 'norm', 'dZ_ev', 'dR_ev', 'dZdZ_ev', 'dRdR_ev',\
            'dZdR_ev', 'dZ_perc', 'dR_perc', 'dZdZ_perc', 'dRdR_perc',\
            'dZdR_perc', 'representation_form']
    
    def __init__(self, filename, Z, M = None):
        '''
        filename: string
        Z: np array, nuclear charges
        M: np tuple, represented compound
        '''

        self.filename = filename
        self.Z = Z
        if not (M == None):
            self.norm = jnp.linalg.norm(M, ord = 'nuc')
        
        self.dZ_ev = None
        self.dR_ev = None
        self.dZdZ_ev = None
        self.dRdR_ev = None
        self.dZdR_ev = None

        self.dZ_perc = None
        self.dR_perc = None
        self.dZdZ_perc = None
        self.dRdR_perc = None
        self.dZdR_perc = None

        self.representation_form  = 0

    
    def add_all_RZev(self, dZ, dR, dZdZ, dRdR, dZdR):
           
        self.representation_form = dZ[0].size
        print(self.representation_form)

        dZ_ev = []
        dR_ev = []

        dZdZ_ev = []
        dZdR_ev = []
        dRdR_ev = []
        
        dim = len(self.Z)
        #for matrix like representations, calculate eigenvalues. for the rest, leave as it is


        matrix = False
        try:
            if(dZ[0].shape[0] == dZ[0].shape[1]):
                matrix = True
        except IndexError:
            print("your representation has not the shape of a matrix")

        if matrix:            
            for i in range(dim):
                print(dZ[i].shape)
                eigenvals, eigenvec = jnp.linalg.eig(dZ[i])
                dZ_ev.append(eigenvals)

                for x in range(3):
                    eigenvals, eigenvec = jnp.linalg.eig(dR[i,x])
                    dR_ev.append(eigenvals)

                    for j in range(dim):
                        eigenvals, eigenvec = jnp.linalg.eig(dZdR[i, j, x])
                        dZdR_ev.append(eigenvals)

                        for y in range(3):
                            eigenvals, eigenvec = jnp.linalg.eig(dRdR[i, x, j, y])
                            dRdR_ev.append(eigenvals)

                for j in range(dim):
                    eigenvals, eigenvec = jnp.linalg.eig(dZdZ[i,j])
                    dZdZ_ev.append(eigenvals)
        
        else:
            #print("dZdR size:", len(dZdR), len(dZdR[0]), len(dZdR[0][0]))
            for i in range(dim):
                dZ_ev.append(dZ[i])
                for x in range(3):
                    dR_ev.append(dR[i,x])
                    for j in range(dim):
                        dZdR_ev.append(dZdR[i,j, x])
                        for y in range(3):
                            dRdR_ev.append(dRdR[i,x,j,y])
                for j in range(dim):
                    dZdZ_ev.append(dZdZ[i,j])
        
        self.dZ_ev = np.asarray(dZ_ev)
        self.dR_ev = np.asarray(dR_ev)
        self.dZdZ_ev = np.asarray(dZdZ_ev)
        self.dRdR_ev = np.asarray(dRdR_ev)
        self.dZdR_ev = np.asarray(dZdR_ev)
    
    def convert_ev_to_np(self):
        self.dZ_ev = np.asarray(self.dZ_ev)
        self.dR_ev = np.asarray(self.dR_ev)
        self.dZdZ_ev = np.asarray(self.dZdZ_ev)
        self.dRdR_ev = np.asarray(self.dRdR_ev)
        self.dZdR_ev = np.asarray(self.dZdR_ev)

    
    def calculate_percentage(self, repro):
        '''
        Calculates fraction of nonzero values compared to full dim of fingerprint
        repro: int, defines which representation was used:
                0 = CM, 1 = EVCM, 2 = BoB, 3 = OM, 4  = EVOM
        '''
        
        N = len(self.Z)
        OM_dim = jrep.OM_dimension(self.Z)
        BoB_dim = jrep.BoB_dimension(self.Z)
        dimension = [N*N, N, BoB_dim, OM_dim*OM_dim, OM_dim]

        dim = dimension[repro]


        try:
            self.dZ_perc = np.count_nonzero(np.asarray(self.dZ_ev))/(dim*N) #is 2*dim Z the max number of EV?
            self.dR_perc = np.count_nonzero(np.asarray(self.dR_ev))/(dim*3*N)
            self.dZdZ_perc = np.count_nonzero(np.asarray(self.dZdZ_ev))/(dim*N*N)
            self.dRdR_perc = np.count_nonzero(np.asarray(self.dRdR_ev))/(9*dim*N*N)
            self.dZdR_perc = np.count_nonzero(np.asarray(self.dZdR_ev))/(3*dim*N*N)

        except ValueError:
            print("an error occured while calculating percentage")

        fractions = [self.dZ_perc, self.dR_perc, self.dZdZ_perc, self.dRdR_perc, self.dZdR_perc]

        return(fractions)

    def calculate_dim(self, repro):
        '''
        repro: int, defines which representation was used:
                0 = CM, 1 = EVCM, 2 = BoB, 3 = OM, 4  = EVOM
        '''
        N = len(self.Z)
        OM_dim = jrep.OM_dimension(self.Z)
        BoB_dim = jrep.BoB_dimension(self.Z)
        dimension = [N*N, N, BoB_dim, OM_dim*OM_dim, OM_dim]

        return(dimension[repro])


        

    
    def transfer_to_numpy(self, listofself):
        for i in listofself:
            i = np.asarray(i)

        return(listofself)


    def calculate_smallerthan(self, repro = 0, lower_bound = 0.00000001):
        '''
        Calculates fraction of nonzero values compared to full dim of fingerprint
        repro: int, defines which representation was used:
                0 = CM, 1 = EVCM, 2 = BoB, 3 = OM, 4  = EVOM
        '''
        
        N = len(self.Z)
        OM_dim = jrep.OM_dimension(self.Z)
        BoB_dim = jrep.BoB_dimension(self.Z)
        dimension = [N*N, N, BoB_dim, OM_dim*OM_dim, OM_dim]
        
        dim = N

        self.representation_form = dimension[repro]
        
        while True: #this is a fix for damaged results files
            try:
                dZ_bigger = self.dZ_ev[(-lower_bound > self.dZ_ev) | (self.dZ_ev > lower_bound)]
                dR_bigger = self.dR_ev[(-lower_bound > self.dR_ev) | (self.dR_ev > lower_bound)]
                dZdZ_bigger = self.dZdZ_ev[(-lower_bound > self.dZdZ_ev) | (self.dZdZ_ev > lower_bound)]
                dRdR_bigger = self.dRdR_ev[(-lower_bound > self.dRdR_ev) | (self.dRdR_ev > lower_bound)]
                dZdR_bigger = self.dZdR_ev[(-lower_bound > self.dZdR_ev) | (self.dZdR_ev > lower_bound)]
                
                break

            except ValueError:
                print("an error occuerd while calculating ev values smaller than ", lower_bound)
                break
            except TypeError:
                self.dZ_ev = np.asarray(self.dZ_ev)
                self.dR_ev = np.asarray(self.dR_ev)
                self.dZdZ_ev = np.asarray(self.dZdZ_ev)
                self.dRdR_ev = np.asarray(self.dRdR_ev)
                self.dZdR_ev = np.asarray(self.dZdR_ev)

                continue

        #print("self.representation_form", self.representation_form, "dim:", dim)
        #print("size:", self.dZdZ_bigger.size, "dimension foreseen:", self.representation_form*dim**2)
        self.dZ_perc = (len(dZ_bigger))/(self.representation_form*dim) #is 2*dim Z the max number of EV?
        self.dR_perc = (len(dR_bigger))/(3*dim*self.representation_form)
        self.dZdZ_perc = (len(dZdZ_bigger))/(self.representation_form*dim**2)
        self.dRdR_perc = (len(dRdR_bigger))/(self.representation_form*9*dim**2)
        self.dZdR_perc = (len(dZdR_bigger))/(self.representation_form*3*dim**2)

        fractions = [self.dZ_perc, self.dR_perc, self.dZdZ_perc, self.dRdR_perc, self.dZdR_perc]
        numbers = [dZ_bigger, dR_bigger, dZdZ_bigger, dRdR_bigger, dZdR_bigger]

        
        return(fractions, numbers)

class Kernel_Result():
    """ The kernel_result class stores information of kernel results
        
        :param exact_results, test_results: initialize
            the "kernel_result" with exact and test results
        :type exact_results, test_results: numpy.array
        :param sigma, lambda: type float
    """

    __slots__ = ['sigma', 'lamda', 'x', 'y',\
            'x_training', 'x_test', 'y_training', 'y_test',\
        'test_predicted_results', 'mae', 'representation_name',\
        'test_indices', 'training_indices', 'full_kernel_matrix']

    def __init__(self):

        
        empty_array = np.asarray([], dtype = float)
        
        #hyperparameters>c
        self.sigma = float("nan")
        self.lamda = float("nan")
        
        #all data
        self.x = empty_array
        self.y = empty_array

        #training and test set
        self.x_training = empty_array
        self.x_test = empty_array
        
        self.y_training = empty_array
        self.y_test = empty_array
        
        #results
        self.test_predicted_results = empty_array

        self.mae = float("nan")

        #additional information
        self.representation_name = None

        self.test_indices = empty_array
        self.training_indices = empty_array

        self.full_kernel_matrix = None


    def add_results(self,\
            sigma,\
            lamda,\
            y_test = np.asarray([], dtype = float),\
            y_predicted = np.asarray([], dtype = float),\
            mae = float("nan")):

        self.sigma = sigma
        self.lamda = lamda
        
        self.y_test = y_test
        self.test_predicted_results = y_predicted

        if self.mae is float("nan"):
            self.calculate_mae()

    def do_qml_gaussian_kernel(self):
        
        #K is also a np array, create kernel matrix
        K = gaussian_kernel(self.x_training, self.x_training, self.sigma)

        #add small lambda to the diagonal of the kernel matrix
        K[np.diag_indices_from(K)] += self.lamda
        
        #use the built in Cholesky-decomposition to solve
        alpha = cho_solve(K, self.y_training)

        #predict new, calculate kernel matrix between test and training
        Ks = gaussian_kernel(self.x_test, self.x_training, self.sigma)

        #make prediction
        Y_predicted = np.dot(Ks, alpha)

        # Calculate mean-absolute-error (MAE):
        self.mae = np.mean(np.abs(Y_predicted - self.y_test)) 
        self.test_predicted_results = Y_predicted

    def laplacian_kernel_matrix(self, x_training, x_test):

        #create full laplacian kernel matrix
        K = laplacian_kernel(x_test, x_training, self.sigma)

        self.full_kernel_matrix = K

        return(K)
    
    def calculate_mae(self):
        """Calculates mean average error
        Between exact and test result
        """
        self.mae = np.mean(np.abs(self.y_test - self.test_predicted_results))
        return(self.mae)

    def result_name(self):
        return("%s_l%f_s%f" %(self.representation_name, self.lamda, self.sigma))

    def training_size(self):
        try:
            return(len(self.y_training))
        except Error:
            return(len(self.x_training))


class CurveObj:

    __slots__ = ['xnparray', 'ynparray', 'xerror', 'yerror', 'name']

    def __init__(self, name):

        empty_array = np.asarray([], dtype = float)

        self.xnparray = empty_array
        self.ynparray = empty_array
        self.xerror = None
        self.yerror = None
        self.name = name



def read_xyz_energies(folder, get_energy = True):
    ''' Opens files in xyz folder and stores Z, R and N data + makes representation
    Variables
    ---------
    folder: Path to folder containing xyz files
    representation: some kind of representation that can work with Z, R, N data
    Returns
    -------
    compound_list: List of information on files in file_list
                    [name of file, Z, R, N]
                    with Z, R and N being numpy arrays and the
                    name of file being a string
    '''

    molecule_list = []
    compound_ls = []

    for xyzfile in os.listdir(folder):
        if xyzfile.endswith(".xyz"):
            filename = folder + xyzfile
            atoms = []
            R = []



            #open file and read lines
            with open(filename, 'r') as f:
                content = f.readlines()
        
            N = int(content[0])

            if get_energy:
                '''in the QM9 database, the comment line contains information on the molecule.
                See https://www.nature.com/articles/sdata201422/tables/4 for more info.
                '''
                try:
                    comment = content[1].split()
                    #extract internal energy at 0K in Hartree
                    zero_point_energy = comment[12]
                except IndexError:
                    print("The xyz file does not support energy value passing analogous to the QM9 database \n no energy retrieved")
                    get_energy = False
                

            #read in atomic information from xyz file
            for line in range(2, N+2):
                atominfo = content[line].split()
                atoms.append(atominfo[0])
                try:
                    R.append(jnp.asarray([float(atominfo[1]), float(atominfo[2]), float(atominfo[3])]))
                except ValueError:
                    '''in the QM9 dataset, some values are not in scientific notation
                    they cause errors when reading, this ValueError deals with them
                    '''
                    print("a Value Error occured while reading file %s. The following line caused Errors:" %xyzfile)
                    print(atominfo)
                    for i in range(1, 4):
                        atominfo[i] = atominfo[i].replace("*^","e") 
                    R.append(jnp.asarray([float(atominfo[1]), float(atominfo[2]), float(atominfo[3])]))

            #transform to np arrays for further use
            Z = np.asarray([float(jbas.atomic_signs[atom]) for atom in atoms])
            R = np.asarray(R)
            
            #create list of compound information and represented information
            c = compound(xyzfile, Z, R, N)
            ha = c.heavy_atoms()

            if get_energy:
                c.add_energy(zero_point_energy)
                molecule_list.append([xyzfile,Z, R, N, atoms, ha, zero_point_energy])
            else:
                molecule_list.append([xyzfile,Z, R, N, atoms, ha])

            compound_ls.append(c)

    return(molecule_list, compound_ls)


def read_xyzfile(filename, get_energy = False):
    '''
    reads single file, returns Z

    '''
    X = []
    atoms = []
    zero_point_energy = 0

    #open file and read lines
    with open(filename, 'r') as f:
        content = f.readlines()
        N = int(content[0])

        if get_energy:
            '''in the QM9 database, the comment line contains information on the molecule.
            See https://www.nature.com/articles/sdata201422/tables/4 for more info.
            '''
            try:
                comment = content[1].split()
                #extract internal energy at 0K in Hartree
                zero_point_energy = comment[12]
            except IndexError:
                print("The xyz file does not support energy value passing analogous to the QM9 database \n no energy retrieved")
                get_energy = False
        
        #read in atomic information from xyz file
        for line in range(2, N+2):
            atominfo = content[line].split()
            atoms.append(atominfo[0])
            try:
                X.append(jnp.asarray([float(atominfo[1]), float(atominfo[2]), float(atominfo[3])]))
            except ValueError:
                '''in the QM9 dataset, some values are not in scientific notation
                they cause errors when reading, this ValueError deals with them
                '''
                print("a Value Error occured while reading file %s. The following line caused Errors:" %xyzfile)
                print(atominfo)
                for i in range(1, 4):
                    atominfo[i] = atominfo[i].replace("*^","e")
                X.append(jnp.asarray([float(atominfo[1]), float(atominfo[2]), float(atominfo[3])]))

            #transform to np arrays for further use
            Z = jnp.asarray([jbas.atomic_signs[atom] for atom in atoms])
            R = jnp.asarray(X)

    return(Z, R, N, zero_point_energy)


def make_compound_instances(molecule_list):
    '''converts molecular information to compound instances'''
    dim = len(molecule_list)
    compound_ls = []
    heavy_atoms_ls = []
    for i in range(dim):
        filename = compound_list[i,0]
        Z = compound_list[i, 1]
        R = compound_list[i, 2]
        N = compound_list[i, 3]
        atoms = compound_list[i, 4]
        energy = compound_list[i, 5]
        c = compound(filename, Z, R, N)
        c.add_energy(energy)
        ha = c.heavy_atoms
        
        heavy_atoms_ls.append(ha)
        compound_ls.append(c)

def store_compounds(compound_list, destination_file):
    '''stores componds to destination_file'''
    with open(destination_file, 'wb') as f:
        pickle.dump(compound_list, f)
    return(print("compounds have been stored to ", destination_file))


def read_compounds(source_file):
    '''reads source file and returns compound_list'''
    with open(source_file, 'rb') as f:
        compound_list = pickle.load(f)

    return(compound_list)

def sortby_heavyatoms(source_file, destination_file, heavy_atoms):
    '''retrieves all compounds from source file and saves all atoms that contain
    up to the defined number of heavy atoms
    '''
    all_compounds = read_compounds(source_file)

    ha_compliant_compounds = []
    max_atoms = 0

    for c in all_compounds:
        if (c.heavy_atoms() <= heavy_atoms):
            ha_compliant_compounds.append(c)
            if len(c.Z) > max_atoms:
                max_atoms = len(c.Z)
            

    store_compounds(ha_compliant_compounds, destination_file)
    print("a new compound list containing only files/n with up to %i heavy atoms has been saved to %s" %(heavy_atoms, destination_file))

    return(max_atoms)


def read_xyz_qml(pathway):
    '''function that reads all xyz files in pathway and returns list of Z, R, N information
    input
    -----
    pathway: string, pathway to folder containing '.xyz' files.
            ends with '/'

    output
    ------
    compoundlist: list containing compound information (qml element)
    ZRN_data: list containing Z, R and N arrays of the compounds
    '''
    compoundlist = []
    ZRN_data = []

    print("iterate over all molecules")
    for xyzfile in os.listdir(database):
        xyz_fullpath = database + xyzfile #probably path can be gotten more directly
        compound = qml.Compound(xyz_fullpath)

        print("compound %s" % xyzfile)
        Z = compound.nuclear_charges.astype(float)
        R = compound.coordinates
        N = float(len(Z))

        compoundlist.append(compound)
        ZRN_data.append(Z, R, N)

    return(compoundlist, ZRN_data)

def alter_coordinates(coordinates, d, h):
    Z = copy.deepcopy(coordinates[0])
    R = copy.deepcopy(coordinates[1])
    N = coordinates[2]
    if d[0] == 0:
        Z[d[1]] += h
        return([Z, R, N])

    if d[0] == 1:
        R[d[1]][d[2]] += h

        return([Z, R, N])


def atomization_energy(potential_energy, nuclear_charges):
    '''calculates atomization energy in kcal/mol based on
    potential energy in hartree and nuclear charges,
    Data taken from Ramakrishnan, Raghunathan; Dral, Pavlo; Dral, Pavlo O.;\\
    Rupp, Matthias; Anatole von Lilienfeld, O. (2014):\\
    Quantum chemistry structures and properties of 134 kilo molecules.\\
    figshare. Collection. https://doi.org/10.6084/m9.figshare.978904
    '''
    #atom energy of single atoms, nuclear charge: U_0 in hartree
    atom_energy_hartree = {1:-0.500273 , 6:-37.846772, 7:-54.583861, 8:-75.064579 , 9:-99.718730}
    
    total_atom_energy = 0

    for Z in nuclear_charges:
        total_atom_energy += atom_energy_hartree[Z]

    atomization_energy_hartree = potential_energy - total_atom_energy
    
    atomization_energy_kcal = atomization_energy_hartree * 627.503

    return(atomization_energy_kcal)

'''in this file some more or less efficient methods will be written to retrieve xyz data from files, store it in one file to make programs more efficient, and to open such a file with a lot of xyz data and reread it.
'''
import pickle
import os
import jax.numpy as jnp
import jax_basis as jbas

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
    def __init__(self, filename, Z, M):
        '''
        filename: string
        Z: np array, nuclear charges
        M: np tuple, represented compound
        '''

        self.filename = filename
        self.Z = Z
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

    
    def add_all_RZev(self, dZ, dR, dZdZ, dRdR, dZdR):
        
        dZ_ev = []
        dR_ev = []

        dZdZ_ev = []
        dZdR_ev = []
        dRdR_ev = []
        
        dim = len(self.Z)

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

        
        self.dZ_ev = dZ_ev
        self.dR_ev = dR_ev
        self.dZdZ_ev = dZdZ_ev
        self.dRdR_ev = dRdR_ev
        self.dZdR_ev = dZdR_ev
    
    def add_Z_norm(self, Z, M):
        self.Z = Z
        self.dim = M.shape[0]
        self.norm = jnp.linalg.norm(M, ord = 'nuc')

    def calculate_percentage(self):
        dim = len(self.Z)
        print("Z:", self.Z, "len Z:", dim)
        try:
            dZdZ_len = jnp.asarray(self.dZdZ_ev).shape[0]
            dZdZ_evno = jnp.count_nonzero(jnp.asarray(self.dZdZ_ev))
            print("dZdZ ev:\n", jnp.asarray(self.dZdZ_ev), "\nlen of dZdZ (shape): ", dZdZ_len, "len of dZdZ calculated:", dim**3, "nonzero EV: ", dZdZ_evno)

            old_dZ_perc = len(self.dZ_ev)/jnp.count_nonzero(jnp.asarray(self.dZ_ev))
            corr_dZ_perc = jnp.count_nonzero(jnp.asarray(self.dZ_ev))/jnp.asarray(self.dZ_ev).shape[0]
            self.dZ_perc = jnp.count_nonzero(jnp.asarray(self.dZ_ev))/(dim**2) #is 2*dim Z the max number of EV?
            
            old_dR_perc = len(self.dR_ev)/jnp.count_nonzero(jnp.asarray(self.dR_ev))
            corr_dR_perc = jnp.count_nonzero(jnp.asarray(self.dR_ev))/ jnp.asarray(self.dZ_ev).shape[0]
            self.dR_perc = jnp.count_nonzero(jnp.asarray(self.dR_ev))/(3*dim**2)
            
            old_dZdZ_perc = len(self.dZdZ_ev)/jnp.count_nonzero(jnp.asarray(self.dZdZ_ev))
            corr_dZdZ_perc = jnp.count_nonzero(jnp.asarray(self.dZdZ_ev))/jnp.asarray(self.dZdZ_ev).shape[0]
            self.dZdZ_perc = jnp.count_nonzero(jnp.asarray(self.dZdZ_ev))/(dim**3)
            
            dRdR_evno = jnp.count_nonzero(jnp.asarray(self.dRdR_ev))
            dRdR_array = jnp.asarray(self.dRdR_ev)
            print("dRdR ev:\n", dRdR_array, "\nnonzero EV: ", dRdR_evno, "calculated nu of max EV: ", 9*dim**3)

            old_dRdR_perc = len(self.dRdR_ev)/jnp.count_nonzero(jnp.asarray(self.dRdR_ev))
            corr_dRdR_perc = jnp.count_nonzero(jnp.asarray(self.dRdR_ev))/jnp.asarray(self.dRdR_ev).shape[0]
            self.dRdR_perc = jnp.count_nonzero(jnp.asarray(self.dRdR_ev))/(9*dim**3)
            
            old_dZdR_perc = len(self.dZdR_ev)/jnp.count_nonzero(jnp.asarray(self.dZdR_ev))
            corr_dZdR_perc = jnp.count_nonzero(jnp.asarray(self.dZdR_ev))/jnp.asarray(self.dZdR_ev).shape[0]
            self.dZdR_perc = jnp.count_nonzero(jnp.asarray(self.dZdR_ev))/(3*dim**3)

        except ValueError:
            print("an error occured while calculating percentage")

        old_values = [old_dZ_perc, old_dR_perc, old_dZdZ_perc, old_dRdR_perc, old_dZdR_perc]
        corrected_values = [corr_dZ_perc, corr_dR_perc, corr_dZdZ_perc, corr_dRdR_perc, corr_dZdR_perc]
        new_values = [self.dZ_perc, self.dR_perc, self.dZdZ_perc, self.dRdR_perc, self.dZdR_perc]

        return(old_values, corrected_values, new_values)
        

            
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
            Z = jnp.asarray([jbas.atomic_signs[atom] for atom in atoms])
            R = jnp.asarray(R)
            
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

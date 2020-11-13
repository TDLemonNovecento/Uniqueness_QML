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
    def __init__(self, filename, Z):
        self.filename = filename
        self.Z = Z
    
    def add_all_RZev(self, dZ_ev, dR_ev, dZdZ_ev, dRdR_ev, dZdR_ev):
        self.dZ_ev = dZ_ev
        self.dR_ev = dR_ev
        self.dRdZ_ev = dZdZ_ev
        self.dRdR_ev = dZdR_ev
        self.dZdR_ev = dZdR_ev

    def calculate_percentage(self):
        self.dZ_perc = 
        self.dR_perc = []
        slef.dZdZ_perc = []
        self.dRdR_perc = []
        self.dZdR_perc = []

        #dZ
        for eigenvals in dZ_ev:
            ev = jnp.real(eigenvals)
            nz = jnp.count_nonzero(ev)
            perc = nz/ max_ev
            self.dZ_perc.append(perc)

        #dR
        for eigenvals in dR_ev:


        def calculate_perc(listed_ev):
            shape = listed_ev.shape
            
            for i in 
            for eigenvals in listed_ev:
                ev = jnp.real(eigenvals)
                nz = jnp.count_nonzero(ev)
                perc = nz/ max_ev
                self.dZ_perc.append(perc)
            return(perc_list)

            
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

    for c in all_compounds:
        if (c.heavy_atoms() <= heavy_atoms):
            ha_compliant_compounds.append(c)

    store_compounds(ha_compliant_compounds, destination_file)
    return(print("a new compound list containing only files/n with up to %i heavy atoms has been saved to %s" %(heavy_atoms, destination_file)))


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


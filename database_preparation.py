'''in this file some more or less efficient methods will be written to retrieve xyz data from files, store it in one file to make programs more efficient, and to open such a file with a lot of xyz data and reread it.
'''
import pickle

class comound():
    '''
    stores information of molecules
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
        unique, counts = jnp.unique(Z, return_counts = True)
        dictionary =dict(zip(unique,counts))
        heavy_atoms = Z.size - dictionary[1]
        return(heavy_atoms)

    

def read_xyz(folder, representation = jrep.CM_eigenvectors_EVsorted, get_energy = False):
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
                comment = content[1].split()
                #extract internal energy at 0K in Hartree
                zero_point_energy = comment[12]
            else:
                zero_point_energy = 0

            #read in atomic information from xyz file
            for line in range(2, N+2):
                atominfo = content[line].split()
                atoms.append(atominfo[0])
                try:
                    R.append(np.array([float(atominfo[1]), float(atominfo[2]), float(atominfo[3])]))
                except ValueError:
                    '''in the QM9 dataset, some values are not in scientific notation
                    they cause errors when reading, this ValueError deals with them
                    '''
                    print("a Value Error occured while reading the xyz file. The following line caused Errors:")
                    print(atominfo)
                    for i in range(1, 4):
                        atominfo[i] = atominfo[i].replace("*^","e") 
                    R.append(np.array([float(atominfo[1]), float(atominfo[2]), float(atominfo[3])]))

            #transform to np arrays for further use
            Z = np.array([atomic_signs[atom] for atom in atoms])
            R = np.array(R)
            
            #create list of compound information and represented information
            molecule_list.append([xyzfile,Z, R, N, atoms, zero_point_energy])

    return(molecule_list)



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


def read_all_xyz(pathway):
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


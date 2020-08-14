#contains representation mapping functions
import qml


def cm(xyz):
    #coulomb matrix
    size = 2 
    mol = qml.Compound(xyz)
    mol.generate_coulomb_matrix(size = size, sorting="row-norm")
    return(mol.representation)

def cm_ev(xyz):
    mol = qml.Compound(xyz)
    mol.generate_eigenvalue_coulomb_matrix(mol)
    return(mol.representation)

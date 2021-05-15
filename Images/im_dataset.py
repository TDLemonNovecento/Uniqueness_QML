import sys
sys.path.insert(0, "..")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import database_preparation as datprep
from collections import Counter

#plotting stuff
fontsize = 30

plt.rc('font',       size=fontsize) # controls default text sizes
plt.rc('axes',  titlesize=fontsize) # fontsize of the axes title
plt.rc('axes',  labelsize=fontsize) # fontsize of the x and y labels
plt.rc('xtick', labelsize=fontsize*0.8) # fontsize of the tick labels
plt.rc('ytick', labelsize=fontsize*0.8) # fontsize of the tick labels
plt.rc('legend', fontsize=fontsize*0.8) # legend fontsize
plt.rc('figure',titlesize=fontsize*1.2) # fontsize of the figure title
fig = plt.figure(figsize = (12, 8))

gs = GridSpec(2, 10)
axt1= fig.add_subplot(gs[0,:5])
axt2= fig.add_subplot(gs[0,5:])

axb1= fig.add_subplot(gs[1,0:2])
axb2= fig.add_subplot(gs[1,2:4])
axb3= fig.add_subplot(gs[1,4:6])
axb4= fig.add_subplot(gs[1,6:8])
axb5= fig.add_subplot(gs[1,8:10])



plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=3, hspace=None)


database = "../Databases/Pickled/qm7.pickle"


compoundlist = datprep.read_compounds(database)
#print("length of compoundlist:", len(compoundlist))

Zlist = []
atomlen = []
halist = []
atomtypes = []

atomfrequencylist = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

for compound in compoundlist:
    Z = compound.Z
    atoms = len(Z)
    atomscount = Counter(list(Z))
    atomscountdict = dict(atomscount)
    atomtype = list(atomscountdict.keys())
    no_ofatoms = list(atomscountdict.values())
    

    for i in range(len(atomtype)):
        atomtypes.append(atomtype[i])
        atomfrequencylist[atomtype[i]].append(no_ofatoms[i])
        if atomtype[i] == 9:
            print(compound.filename)
    ha = compound.heavy_atoms()
     
    Zlist.append(Z)
    halist.append(ha)
    atomlen.append(atoms)


ha_mols_counter = Counter(halist) #count how many times an element occurs in halist

H_frequency = Counter(atomfrequencylist[1])
C_frequency = Counter(atomfrequencylist[6])
N_frequency = Counter(atomfrequencylist[7])
O_frequency = Counter(atomfrequencylist[8])
F_frequency = Counter(atomfrequencylist[9])

atomscounter = Counter(atomtypes)
atomtypes_counter = Counter(atomtypes)
print("atomtypes:", atomtypes_counter)

axb1.bar(H_frequency.keys(), H_frequency.values(), color = 'gray', label = 'H')
axb2.bar(C_frequency.keys(), C_frequency.values(), color = 'black', label = 'C')
axb3.bar(N_frequency.keys(), N_frequency.values(), label = 'N')
axb4.bar(O_frequency.keys(), O_frequency.values(), color = 'red', label = 'O')
axb5.bar(F_frequency.keys(), F_frequency.values(), color = 'green', label = 'F')


axt2.bar(ha_mols_counter.keys(), ha_mols_counter.values(), label = "Heavy Atoms in Molecule")
axt1.bar(atomscounter.keys(), atomscounter.values(), label = "Atoms in Molecule")

print("maximum no. of atoms:", max(atomscounter.keys()))
axb1.set_ylabel('Number of Molecules')
axt1.set_xticks(np.arange(0, 12,2))
axt2.set_yticks(np.arange(0, 4000, 2000))



#fig.tight_layout()
plt.show()

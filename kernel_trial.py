import kernel_learning as kler

datapath = "/home/stuke/Databases/QM9_XYZ"

represented, compound = kler.read_xyz([datapath + '/dsgdb9nsd_000001.xyz'])
print(compound)


import database_preparation as datprep

results_file = "./Pickled/trial_numder.pickle"

res = datprep.read_compounds(results_file)
print("results:")
for r in res:
    print(r.filename)
    print(r.dRdR_ev)

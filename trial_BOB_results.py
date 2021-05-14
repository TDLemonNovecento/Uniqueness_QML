import database_preparation2 as datprep

new_filename = "/home/linux-miriam/Databases/Pickled/BOB2_numder_res"

numbers = ["100-800", "100-800"]


res1 = datprep.read_compounds(new_filename + numbers[0])

res2 = datprep.read_compounds(new_filename + numbers[1])

print("res100-200 0, 100:")
print(res1[0].dZ_perc)
print(res1[99].filename)

print("res200-300 0, 100:")
print(res2[100].filename)
print(res2[199].filename)

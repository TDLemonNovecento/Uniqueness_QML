'''this file contains basis functions for any representations that rely on them'''

'''the following dictionary contains the orbital basis
for the respective nuclear charge of the atoms from H to Ca
'''
orbital_configuration = {
        1:['1s'],
        2:['1s'],
        3:['1s','2s','2px','2py','2pz'],
        4:['1s','2s','2px','2py','2pz'],
        5:['1s','2s','2px','2py','2pz'],
        6:['1s','2s','2px','2py','2pz'],
        7:['1s','2s','2px','2py','2pz'],
        8:['1s','2s','2px','2py','2pz'],
        9:['1s','2s','2px','2py','2pz'],
        10:['1s','2s','2px','2py','2pz'],
        11:['1s','2s','2px','2py','2pz','3s','3px','3py','3pz'],
        12:['1s','2s','2px','2py','2pz','3s','3px','3py','3pz'],
        13:['1s','2s','2px','2py','2pz','3s','3px','3py','3pz'],
        14:['1s','2s','2px','2py','2pz','3s','3px','3py','3pz'],
        15:['1s','2s','2px','2py','2pz','3s','3px','3py','3pz'],
        16:['1s','2s','2px','2py','2pz','3s','3px','3py','3pz'],
        17:['1s','2s','2px','2py','2pz','3s','3px','3py','3pz'],
        18:['1s','2s','2px','2py','2pz','3s','3px','3py','3pz'],
        19:['1s','2s','2px','2py','2pz','3s','3px','3py','3pz','4s', '4px', '4py', '4pz'],
        20:['1s','2s','2px','2py','2pz','3s','3px','3py','3pz','4s', '4px', '4py', '4pz'],
}

# STO-3G contraction coefficients pulled from EMSL; applies to all atoms
d = [[0.4446345422, 0.5353281423, 0.1543289673],        #contraction coeff. for 1s orbital
        [0.7001154689, 0.3995128261, -0.09996722919],   #contraction coeff. for 2s orbital
        [0.3919573931, 0.6076837186, 0.1559162750],     #contraction coeff. for 2p orbital
        [0.9003984260, 0.2255954336, -0.2196203690],    #contraction coeff. for 3s orbital
        [0.4620010120, 0.55951670053, 0.01058760429],   #contraction coeff. for 3p orbital
        [1.131034442, 0.01960641165, -0.3088441214],    #contraction coeff. for 4s orbital
        [0.5498949471, 0.5715227604, -0.1215468600]]    #contraction coeff. for 4p orbital


a = [
        [
            [0.1688554040, 0.1688554040, 3.425250914],   # H 1s
            [0.0000000000, 0.0000000000, 0.0000000000],  # H 2s,2p
            [0.0000000000, 0.0000000000, 0.0000000000],  # H 3s,3p
            [0.0000000000, 0.0000000000, 0.0000000000]   # H 4s,4p
            ],
        [
            [0.3136497915, 1.158922999, 6.362421394],    # He 1s
            [0.0000000000, 0.0000000000, 0.0000000000],  # He 2s,2p
            [0.0000000000, 0.0000000000, 0.0000000000],  # He 3s,3p
            [0.0000000000, 0.0000000000, 0.0000000000]   # He 4s,4p
            ],
        [
            [0.7946504870, 2.936200663, 16.11957475],    # Li 1s
            [0.04808867840, 0.1478600533, 0.6362897469], # Li 2s,2p
            [0.0000000000, 0.0000000000, 0.0000000000],  # Li 3s,3p
            [0.0000000000, 0.0000000000, 0.0000000000]   # Li 4s,4p
            ],
        [
            [1.487192653, 5.495115306, 30.16787069],     # Be 1s
            [0.09937074560, 0.3055389383, 1.314833110],  # Be 2s,2p
            [0.0000000000, 0.0000000000, 0.0000000000],  # Be 3s,3p
            [0.0000000000, 0.0000000000, 0.0000000000]   # Be 4s,4p
            ],
        [
            [2.405267040, 8.887362172, 48.79111318],     # B 1s
            [0.1690617600, 0.5198204999, 2.236956142],   # B 2s,2p
            [0.0000000000, 0.0000000000, 0.0000000000],  # B 3s,3p
            [0.0000000000, 0.0000000000, 0.0000000000]   # B 4s,4p
            ],
        [
            [3.530512160, 13.04509632, 71.61683735],     # C 1s
            [0.2222899159, 0.6834830964, 2.941249355],   # C 2s,2p
            [0.0000000000, 0.0000000000, 0.0000000000],  # C 3s,3p
            [0.0000000000, 0.0000000000, 0.0000000000]   # C 4s,4p
            ],
        [
            [8.85660238, 18.05231239, 99.10616896],      # N 1s
            [0.2857143744, 0.8784966449, 3.780455879],   # N 2s,2p
            [0.0000000000, 0.0000000000, 0.0000000000],  # N 3s,3p
            [0.0000000000, 0.0000000000, 0.0000000000]   # N 4s,4p
            ],
        [
            [6.443608313, 23.80886605, 130.7093214],     # O 1s
            [0.3803889600, 1.169596125, 5.033151319],    # O 2s,2p
            [0.0000000000, 0.0000000000, 0.0000000000],  # O 3s,3p
            [0.0000000000, 0.0000000000, 0.0000000000]   # O 4s,4p
            ],
        [
            [8.216820672, 30.36081233, 166.6791340],     # F 1s
            [0.4885884864, 1.502281245, 6.464803249],    # F 2s,2p
            [0.0000000000, 0.0000000000, 0.0000000000],  # F 3s,3p
            [0.0000000000, 0.0000000000, 0.0000000000]   # F 4s,4p
            ],
        [
            [10.20529731, 37.70815124, 207.0156070],     # Ne 1s
            [0.6232292721, 1.916266291, 8.246315120],    # Ne 2s,2p
            [0.0000000000, 0.0000000000, 0.0000000000],  # Ne 3s,3p
            [0.0000000000, 0.0000000000, 0.0000000000]   # Ne 4s,4p
            ],
        [
            [12.36238776, 45.67851117, 250.7724300],     # Na 1s
            [0.9099580170, 2.797881859, 12.04019274],    # Na 2s,2p
            [0.1614750979, 0.4125648801, 1.478740622],    # Na 3s,3p
            [0.0000000000, 0.0000000000, 0.0000000000]   # Na 4s,4p
            ],
        [
            [14.75157752, 54.50646845, 299.2374137],     # Mg 1s
            [1.142857498, 3.513986579, 15.12182352],     # Mg 2s,2p
            [0.1523797659, 0.3893265318, 1.395448293],   # Mg 3s,3p
            [0.0000000000, 0.0000000000, 0.0000000000]   # Mg 4s,4p
            ],
        [
            [17.32410761, 64.01186067, 351.4214767],     # Al 1s
            [1.428353970, 4.391813233, 18.89939621],     # Al 2s,2p
            [0.1523797659, 0.3893265318, 1.395448293],   # Al 3s,3p
            [0.0000000000, 0.0000000000, 0.0000000000]   # Al 4s,4p
            ],
        [
            [20.10329229, 74.28083305, 407.7975514],     # Si 1s
            [1.752899952, 5.389706871, 23.19365606],     # Si 2s,2p
            [0.1614750979, 0.4125648801, 1.478740622],   # Si 3s,3p
            [0.0000000000, 0.0000000000, 0.0000000000]   # Si 4s,4p
            ],
        [
            [23.08913156, 85.31338559, 468.3656378],     # P 1s
            [2.118614352, 6.514182577, 28.03263958],     # P 2s,2p
            [0.1903428909, 0.4863213771, 1.743103231],   # P 3s,3p
            [0.0000000000, 0.0000000000, 0.0000000000]   # P 4s,4p
            ],
        [
            [26.28162542, 97.10951830, 533.1257359],     # S 1s
            [2.518952599, 7.745117521, 33.32975173],     # S 2s,2p
            [0.2215833792, 0.5661400518, 2.029194274],   # S 3s,3p
            [0.0000000000, 0.0000000000, 0.0000000000]   # S 4s,4p
            ],
        [
            [29.64467686, 109.5358542, 601.3456136],     # Cl 1s
            [2.944499834, 9.053563477, 38.96041889],     # Cl 2s,2p
            [0.2325241410, 0.5940934274, 2.129386495],   # Cl 3s,3p
            [0.0000000000, 0.0000000000, 0.0000000000]   # Cl 4s,4p
            ],
        [
            [33.24834945, 122.8512753, 674.4465184],     # Ar 1s
            [3.413364448, 10.49519900, 45.16424392],     # Ar 2s,2p
            [0.2862472356, 0.7313546050, 2.621366518],   # Ar 3s,3p
            [0.0000000000, 0.0000000000, 0.0000000000]   # Ar 4s,4p
            ],
        [
            [38.03332899, 140.5315766, 771.5103681],     # K 1s
            [3.960373165, 12.17710710, 52.40203979],     # K 2s,2p
            [0.3987446295, 1.018782663, 3.651583985],    # K 3s,3p
            [0.08214006743, 0.1860011465, 0.5039822505]  # K 4s,4p
            ],
        [
            [42.10144179, 155.5630851, 854.0324951],     # Ca 1s
            [4.501370797, 13.84053270, 59.56029944],     # Ca 2s,2p
            [0.4777079296, 1.220531941, 4.374706256],    # Ca 3s,3p
            [0.07429520696, 0.1682369410, 0.4558489757]  # Ca 4s,4p
            ]
        ]





sto3Gbasis = [] # instantiate the array representing MO 3G basis

def build_sto3Gbasis(Z, R):
    '''
    Variables
    ---------
    Z : array
        contains nuclear charges pf atp,s
    R : tuple
        contains vectors of atoms position in cartesian coordinates [x,y,z]
    


    Returns
    -------
    sto3Gbasis, dimensions of OM matrix
    '''
    K = 0 # instantiates atomic orbital counter (matrix dimension)

    #loop through atoms array and append orbitals (dictionaries) to the basis array
    #each atom (nuclear number in Z) has an orbital configuration associated, append
    for i in range(len(Z)):
        #print(Z)
        nuc = Z[i]
        #l, m, n correspond to components in slater-type orbitals
        #phi = N *(x**l)*(y**m)*(z**n)*exp(-apha*r)
        #where alpha is the exponential factor
        print("STO3GBasis under construction:")
        #print("Atom:", i, "R:", R[i])
        nuc_charge = int(nuc)
        orbitalarray = orbital_configuration[nuc_charge]
        print(orbitalarray) 
        for orbital in orbitalarray:
            if orbital == '1s':
                sto3Gbasis.append(
                    {
                        'Z': nuc_charge,    #atomic number
                        'o': orbital,       #string('1s')
                        'r': R[i],          #position
                        'l': 0,             #angular x momentum number
                        'm': 0,             #angular y momentum number
                        'n': 0,             #angular z momentum number
                        'a': a[ int(nuc_charge) - 1][0],  # get atom specific 1s orbital exp factors
                        'd': d[0]           # get 1s orbital contraction coeff.
                    }
                    )
                K += 1 #increase orbital counter by one
            elif orbital == '2s':
                sto3Gbasis.append(
                    {
                        'Z': nuc_charge,    
                        'o': orbital,       
                        'r': R[i],          
                        'l': 0,             #angular x momentum number
                        'm': 0,             #angular y momentum number
                        'n': 0,             #angular z momentum number
                        'a': a[ nuc_charge - 1][1],  # get atom specific 2s,2p orbital exp factors
                        'd': d[1]           # get 2s orbital contraction coeff.
                    }
                    )
                K += 1 #increase orbital counter by one
            elif orbital == '2px':
                sto3Gbasis.append(
                    {
                        'Z': nuc_charge,
                        'o': orbital,   
                        'r': R[i],      
                        'l': 1,             #angular x momentum number
                        'm': 0,             #angular y momentum number
                        'n': 0,             #angular z momentum number
                        'a': a[ nuc_charge - 1][1],  # get atom specific 2s,2p orbital exp factors
                        'd': d[2]           # get 2p orbital contraction coeff.
                    }
                    )
                K += 1 #increase orbital counter by one
            elif orbital == '2py':
                sto3Gbasis.append(
                    {
                        'Z': nuc_charge,
                        'o': orbital,   
                        'r': R[i],      
                        'l': 0,             #angular x momentum number
                        'm': 1,             #angular y momentum number
                        'n': 0,             #angular z momentum number
                        'a': a[ nuc_charge - 1][1],  # get atom specific 2s,2p orbital exp factors
                        'd': d[2]           # get 2p orbital contraction coeff.
                    }
                    )
                K += 1 #increase orbital counter by one
            elif orbital == '2pz':
                sto3Gbasis.append(
                    {
                        'Z': nuc_charge,    
                        'o': orbital,       
                        'r': R[i],          
                        'l': 0,             #angular x momentum number
                        'm': 0,             #angular y momentum number
                        'n': 1,             #angular z momentum number
                        'a': a[ nuc_charge - 1][1],  # get atom specific 2s,2p orbital exp factors
                        'd': d[2]           # get 2p orbital contraction coeff.
                    }
                    )
                K += 1 #increase orbital counter by one
            elif orbital == '3s':
                sto3Gbasis.append(
                    {
                        'Z': nuc_charge,
                        'o': orbital,   
                        'r': R[i],      
                        'l': 0,             #angular x momentum number
                        'm': 0,             #angular y momentum number
                        'n': 0,             #angular z momentum number
                        'a': a[ nuc_charge - 1][2],  # get atom specific 3s,3p orbital exp factors
                        'd': d[3]           # get 3s orbital contraction coeff.
                    }
                    )
                K += 1 #increase orbital counter by one
            elif orbital == '3px':
                sto3Gbasis.append(
                    {
                        'Z': nuc_charge,
                        'o': orbital,
                        'r': R[i],
                        'l': 1,             #angular x momentum number
                        'm': 0,             #angular y momentum number
                        'n': 0,             #angular z momentum number
                        'a': a[ nuc_charge - 1][2],  # get atom specific 3s,3p orbital exp factors
                        'd': d[4]           # get 3s orbital contraction coeff.
                    }
                    )
                K += 1 #increase orbital counter by one
            elif orbital == '3py':
                sto3Gbasis.append(
                    {
                        'Z': nuc_charge,
                        'o': orbital,
                        'r': R[i],
                        'l': 0,             #angular x momentum number
                        'm': 1,             #angular y momentum number
                        'n': 0,             #angular z momentum number
                        'a': a[ nuc_charge - 1][2],  # get atom specific 3s,3p orbital exp factors
                        'd': d[4]           # get 3s orbital contraction coeff.
                    }
                    )
                K += 1 #increase orbital counter by one
            elif orbital == '3pz':
                sto3Gbasis.append(
                    {
                        'Z': nuc_charge,
                        'o': orbital,
                        'r': R[i],
                        'l': 0,             #angular x momentum number
                        'm': 0,             #angular y momentum number
                        'n': 1,             #angular z momentum number
                        'a': a[ nuc_charge - 1][2],  # get atom specific 3s,3p orbital exp factors
                        'd': d[4]           # get 3s orbital contraction coeff.
                    }
                    )
                K += 1 #increase orbital counter by one
            elif orbital == '4s':
                sto3Gbasis.append(
                    {
                        'Z': nuc_charge,
                        'o': orbital,
                        'r': R[i],
                        'l': 0,             #angular x momentum number
                        'm': 0,             #angular y momentum number
                        'n': 0,             #angular z momentum number
                        'a': a[ nuc_charge - 1][3],  # get atom specific 4s,4p orbital exp factors
                        'd': d[5]           # get 4s orbital contraction coeff.
                    }
                    )
                K += 1 #increase orbital counter by one
            elif orbital == '4px':
                sto3Gbasis.append(
                    {
                        'Z': nuc_charge,
                        'o': orbital,
                        'r': R[i],
                        'l': 1,             #angular x momentum number
                        'm': 0,             #angular y momentum number
                        'n': 0,             #angular z momentum number
                        'a': a[ nuc_charge - 1][3],  # get atom specific 4s,4p orbital exp factors
                        'd': d[6]           # get 4p orbital contraction coeff.
                    }
                    )
                K += 1 #increase orbital counter by one
            elif orbital == '4py':
                sto3Gbasis.append(
                    {
                        'Z': nuc_charge,
                        'o': orbital,
                        'r': R[i],
                        'l': 0,             #angular x momentum number
                        'm': 1,             #angular y momentum number
                        'n': 0,             #angular z momentum number
                        'a': a[ nuc_charge - 1][3],  # get atom specific 4s,4p orbital exp factors
                        'd': d[6]           # get 4p orbital contraction coeff.
                    }
                    )
                K += 1 #increase orbital counter by one
            elif orbital == '4pz':
                sto3Gbasis.append(
                    {
                        'Z': nuc_charge,
                        'o': orbital,
                        'r': R[i],
                        'l': 0,             #angular x momentum number
                        'm': 0,             #angular y momentum number
                        'n': 1,             #angular z momentum number
                        'a': a[ nuc_charge - 1][3],  # get atom specific 4s,4p orbital exp factors
                        'd': d[6]           # get 4p orbital contraction coeff.
                    }
                    )
                K += 1 #increase orbital counter by one
            else:
                try:
                    raise ValueError('your orbital is out of bounds')
                    raise Exception('Error in sto3Gbasis generation was excepted')
                except Exception as error:
                    print("Error in 'build_stoG3basis' function: " + repr(error))

    return(sto3Gbasis, K)







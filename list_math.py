'''
Get a list of keys from dictionary which has value -1
'''
def BoB_emptyZ(dictionary):
    list_of_keys = list()
    list_of_items = dictionary.items()
    for item  in list_of_items:
        if item[1] == -1:
            list_of_keys.append(item[0])
    return  list_of_keys

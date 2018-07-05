import csv
import numpy as np

def get_data(file_name):
    '''
    Get data from csv file
    '''
    f = open(file_name, 'r')
    rdr = csv.reader(f)
    data = []
    for line in rdr:
        data.append(line)

    np_data = np.array(data)
    #print(np_data.shape)
    np_float_data = np_data.astype(np.float)
    #print(np_float_data)
    f.close()

    return np_float_data

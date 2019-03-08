import csv
import numpy as np

def get_data(file_name):
    """Get data from csv file

    input: .csv file path
    
    output: list (np.float type)
    """
    f = open(file_name, 'r')
    rdr = csv.reader(f)
    data = []
    for line in rdr:
        data.append(line)

    np_data = np.array(data)
    np_float_data = np_data.astype(np.float)
    f.close()

    return np_float_data

def set_data(file_name, write_set):
    f = open(file_name, 'w', newline='')
    wr = csv.writer(f)
    wr.writerows(write_set)

    f.close()

import numpy as np

import os
import re

def create_data_set():

    data_set = []
    label_set = []
    for subdirs, dirs, files in os.walk("./"):
        for fil in files:
            if '.npy' in fil:
                dat = np.load(fil)
                if 'labels' in fil:
                    for data in dat:
                        label_set.append(data);
                else:
                    for data in dat:
                        data_set.append(data)

    return (np.array(data_set), np.array(label_set))

set_of_flowlets = create_data_set()




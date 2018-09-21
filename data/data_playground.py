import numpy as np

import os
import re


#creates the data set
def create_data_set():

    data_set = []
    label_set = []
    for subdirs, dirs, files in os.walk("./data"):
        for fil in files:
            if 'data.npy' in fil:
                dat = np.load("./data/" + fil)
                for data in dat:
                    data_set.append(data)
                date_name = fil[:-8]
                for _, _, lab_fil in os.walk("./label"):
                    for lab in lab_fil:
                        if date_name in lab:
                            lab_dat = np.load("./label/" + lab)
                            for label_data in lab_dat:
                                 label_set.append(label_data)

    return (np.array(data_set), np.array(label_set))

set_of_flowlets = create_data_set()




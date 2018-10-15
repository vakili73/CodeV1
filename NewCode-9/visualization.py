# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 15:04:56 2018

@author: vvaki
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

for base, _, files in os.walk('./feature/'):
    for file in files:
        if file.find('test') != -1:
            data = pd.read_csv(base + file, ',')
            data = data.values
            
            feature = data[:, 1:]
            target = data[:, 0]
            
            plt.figure()
            plt.hold(True)
            for i in np.unique(target):
                plt.scatter(feature[(target == i), 0], feature[(target == i), 1])
            
            plt.xlim(feature[:, 0].min(), feature[:, 0].max())
            plt.ylim(feature[:, 1].min(), feature[:, 1].max())
            
            plt.title(file)
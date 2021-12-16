# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 15:22:04 2021

@author: Simon
"""

import os
import getpass
import platform
import hashlib

def md5hash(array):
    xbytes = array.tobytes()
    return hashlib.shake_256(xbytes).hexdigest(5)

def stim2desc(string):
    num = string.split('/')[-1]
    scene_cat, typ, behaviour, memory = num

    convd = {0: {1: 'man-made', 2:'natural'},
             1: {0: 'new', 1:'old'},
             2: {1: 'hit', 2:'miss', 3:'false alarm', 
                 4:'correct rejection', 9: 'N/A'},
             3: {0: 'remembered', 1: 'forgotten'}}
    

    desc = [convd[i][int(n)] for i, n in enumerate(num)]
    return ', '.join(desc)
        
        
###############################
###USER SPECIFIC CONFIGURATION
###############################
username = getpass.getuser().lower()  # your login name
host     = platform.node().lower()    # the name of this computer
system   = platform.system().lower()  # linux, windows or mac.
home = os.path.expanduser('~')

if username == 'simon' and host=='ess-donatra':
    data_dir = 'z:/EEGManyPipelines/'

if username == 'simon' and host=='desktop-simon':
    data_dir = 'z:/EEGManyPipelines/'
    
if username == 'simon.kern' and host=='zilxap29':
    data_dir = '/home/simon.kern/EMP_data/'  
  
else:
    print('Username {} on host {} with {} has no configuration.\n'.format(username,host,system) + \
    'please set user specific information in config.py')

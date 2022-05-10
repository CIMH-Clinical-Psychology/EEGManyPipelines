# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 15:22:04 2021

This is the global configuration file that stores user-specific information.

The datadirectory should be indicated here.

@author: Simon Kern
"""

import os
import getpass
import platform
import hashlib

def md5hash(array):
    if isinstance(array, str):
        xbytes = array.encode()
    else:
        xbytes = array.tobytes()
    return hashlib.shake_256(xbytes).hexdigest(5)

def stim2desc(string):
    if not isinstance(string, str):
        string = str(string)
    num = string.split('/')[-1]
    scene_cat, typ, behaviour, memory = num

    convd = {0: {1: 'man-made', 2:'natural'},
             1: {0: 'new', 1:'old'},
             2: {1: 'hit', 2:'miss', 3:'false-alarm', 
                 4:'correct-rejection', 9: 'N/A'},
             3: {0: 'remembered', 1: 'forgotten', 9:'N/A'}}
    

    desc = [convd[i][int(n)] for i, n in enumerate(num)]
    return '/'.join(desc)    
        
###############################
###USER SPECIFIC CONFIGURATION
###############################
username = getpass.getuser().lower()  # your login name
host     = platform.node().lower()    # the name of this computer
system   = platform.system().lower()  # linux, windows or mac.
home = os.path.expanduser('~')

cache_dir = None

if username == 'simon' and md5hash(host)=='245f2c5750':
    data_dir = 'z:/EEGManyPipelines/'
    cache_dir = 'z:/cache/'
    
elif username == 'simon.kern' and md5hash(host)=='a994e83fa0':
    data_dir = '/home/simon.kern/EMP_data/'  
    cache_dir = '/data/EEGManyPipelines/joblib-cache-2/'
  
else:
    print('Username {} on host {} with {} has no configuration.\n'.format(username,host,system) + \
    'please set user specific information in config.py')

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 15:22:04 2021

@author: Simon
"""

import os
import getpass
import platform



###############################
###USER SPECIFIC CONFIGURATION
###############################
username = getpass.getuser().lower()  # your login name
host     = platform.node().lower()    # the name of this computer
system   = platform.system().lower()  # linux, windows or mac.
home = os.path.expanduser('~')

if username == 'simon' and host=='ess-donatra':
    data_dir = 'z:/EEGManyPipelines/'

elif username == 'simon' and host=='desktop-simon':
    data_dir = 'z:/EEGManyPipelines/'
    
else:
    print('Username {} on host {} with {} has no configuration.\n'.format(username,host,system) + \
    'please set user specific information in config.py')

'''
Created on 15-Jul-2020

@author: user
'''
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import numpy as np
import time
import os
import tensorflow as tf
from konverter import Konverter
from tensorflow.keras.models import load_model
from sys import path
path.append(r"C:\Program Files\casadi-windows-py37-v3.5.5-64bit")
global ca
import casadi as ca

activation_function = 'relu'
root_path = os.path.dirname(os.path.abspath('NeMuGeometry_Approximation.py'))
save_path = root_path + '/Models/NeMu/' + activation_function + '/'

path_NeMu_models = save_path
list_model_names = os.listdir(path_NeMu_models)

for i in range(len(list_model_names)):
    model_name = list_model_names[i]
    if model_name[-2:] == 'h5':
        print(model_name)
        model = load_model(path_NeMu_models + '/' + model_name, compile = False)
        Konverter(model, output_file=save_path + '/' + model_name[:-3] + '_Konverted')# creates the numpy model from the keras model





# list_model_names = os.listdir(path_NeMu_models + '/NumpyModels')
#
#
# index = [idx for idx, s in enumerate(list_model_names) if '.py' in s]
#
# with open(path_NeMu_models + '/NumpyModels/' + list_model_names[0]) as fp:
#     data = fp.read()
#
# for i in index:    # Reading data from file1
#     with open(path_NeMu_models + '/NumpyModels/' + list_model_names[i]) as fp:
#         data_new = fp.read()
#
#     # Merging 2 files
#     # To add the data of file2
#     # from next line
#     data += "\n"
#     data += data_new
#
# with open('NeMu_Numpy.py', 'w') as fp:
#     fp.write(data)
#
#
# from os.path import dirname, basename, isfile, join
# import glob
# modules = glob.glob(join(dirname(path_NeMu_models + '/NumpyModels'), "*.py"))
# __all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
#

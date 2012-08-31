#!/usr/bin/env python
import sys, os, pickle

import flydra_analysis_tools as fat
fad = fat.flydra_analysis_dataset
dac = fat.dataset_analysis_core
tac = fat.trajectory_analysis_core

import numpy as np
import matplotlib
import matplotlib.pyplot

import analysis_modules.odor_packet_analysis as opa

sys.path.append(path)
import analysis_configuration
config = analysis_configuration.Config()


print 'loading culled_dataset, as dataset'
culled_dataset_name = os.path.join(path, config.culled_datasets_path, config.culled_dataset_name)
dataset = fad.load(culled_dataset_name)

print 'loading gaussian odor model, as gm'
gmpath = os.path.join(path, config.odor_gaussian_fit)
fd = open(gmpath, 'r')
gm = pickle.load(fd)
fd.close()

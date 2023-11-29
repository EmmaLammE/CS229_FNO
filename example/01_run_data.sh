#!/bin/bash

# model the seismic data for homogeneous model (500 samples)
jupyter nbconvert --execute --to notebook --allow-errors --inplace data_generation_homo.ipynb

# model the seismic data for layered model (500 samples)
jupyter nbconvert --execute --to notebook --allow-errors --inplace data_generation_layered.ipynb

# model the seismic data for curved model (500 samples)
jupyter nbconvert --execute --to notebook --allow-errors --inplace data_generation_curved.ipynb

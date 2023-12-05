#!/bin/bash
#SBATCH --job-name=train_50
#SBATCH --time=30:00:00
#SBATCH --mem=32GB
#SBATCH -p serc
#SBATCH -c 4
#SBATCH --ntasks=1
#SBATCH --gpus 2
#SBATCH -o ./../sbatch_outputs/out_train_size50.%j.out
#SBATCH -e ./../sbatch_outputs/err_train_size50.%j.err

# below you run/call your code, load modules, python, Matlab, R, etc.
# and do any other scripting you want
# lines that begin with #SBATCH are directives (requests) to the scheduler-SLURM module load python/3.6.1

ml py-numpy/1.24.2_py39
ml py-scipy/1.10.1_py39
ml py-pytorch/2.0.0_py39
ml py-h5py/3.7.0_py39
python3 train_grainsize.py --strainrate_path  ./../data/strainrate_traindata.npy.npz \
                --temperature_path  ./../data/temperature_traindata.npy.npz \
                --grainsize_path  ./../data/grainsize_traindata.npy.npz \
                --save_path ./../model/model_grainsize_ns50_nt10_50 \
                --learn_step 10 \
                --total_step 50

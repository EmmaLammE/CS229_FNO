#!/bin/bash


# Experiment 1: Homogeneous models (500 samples)
# Status: succeed
# python train.py --vp_path  /scr2/haipeng/openfwi_dataset_train/vp_homo_ns500_nx128_nz32.npy \
#                 --vs_path  /scr2/haipeng/openfwi_dataset_train/vs_homo_ns500_nx128_nz32.npy \
#                 --rho_path /scr2/haipeng/openfwi_dataset_train/rho_homo_ns500_nx128_nz32.npy \
#                 --vx_path  /scr2/haipeng/openfwi_dataset_train/vz_homo_ns500_nt801_nx128_nz32.npy \
#                 --save_path /scr2/haipeng/openfwi_dataset_train/model_vz_homo_ns500_nt801_nx128_nz32

# Experiment 2: Layered models (500 samples)
# Status: 
# python train.py --vp_path  /scr2/haipeng/openfwi_dataset_train/vp_layered_ns500_nx128_nz32.npy \
#                 --vs_path  /scr2/haipeng/openfwi_dataset_train/vs_layered_ns500_nx128_nz32.npy \
#                 --rho_path /scr2/haipeng/openfwi_dataset_train/rho_layered_ns500_nx128_nz32.npy \
#                 --vx_path  /scr2/haipeng/openfwi_dataset_train/vz_layered_ns500_nt801_nx128_nz32.npy \
#                 --save_path /scr2/haipeng/openfwi_dataset_train/model_vz_layered_ns500_nt801_nx128_nz32


# Experiment 3: Curved models (500 samples)
# Status: 
python train.py --vp_path  /scr2/haipeng/openfwi_dataset_train/vp_curved_ns500_nx128_nz32.npy \
                --vs_path  /scr2/haipeng/openfwi_dataset_train/vs_curved_ns500_nx128_nz32.npy \
                --rho_path /scr2/haipeng/openfwi_dataset_train/rho_curved_ns500_nx128_nz32.npy \
                --vx_path  /scr2/haipeng/openfwi_dataset_train/vz_curved_ns500_nt801_nx128_nz32.npy \
                --save_path /scr2/haipeng/openfwi_dataset_train/model_vz_curved_ns500_nt801_nx128_nz32

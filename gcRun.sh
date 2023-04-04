#export MODEL_FLAGS="--patch_size 64 --pre_num_channels 32 --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --model_type hierachical"
#export DATA_FLAGS="--data_height 2400 --data_width 3600"
#
#GCDA_MONITOR=1 python3 gc_main.py --command compress --verbose False --model_path ./examples/trained_hierarchical_models/checkpoints/sst-epoch\=008-val_mse_loss\=0.01161-val_loss\=0.07661.pt $MODEL_FLAGS $DATA_FLAGS --input_path data/tccs/ocean/SST_modified/SST.050001-050912.nc --output_path ./outputs/compressed_data  --batch_size 4


export MODEL_FLAGS="--patch_size 64 --pre_num_channels 32 --num_channels 64 --latent_dim 128 --num_embeddings 256 --num_residual_blocks 3 --num_transformer_block 0 --model_type hierachical"
export DATA_FLAGS="--data_height 2400 --data_width 3600"
export TRAIN_FLAGS="--lr 3e-4 --batch_size 1 --epochs 100 --train_verbose True"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/user/abhinand/.conda/envs/py38_13/lib
GCDA_MONITOR=1 python3 gc_main.py --command train --verbose True --data_dir data/tccs/ocean/SST_modified --model_path trained_models/ $MODEL_FLAGS $DATA_FLAGS $TRAIN_FLAGS --resume True --iter -1



keypoint-nerf conda env 
train commmand:
CUDA_VISIBLE_DEVICES=3  python run_nerf_batch.py --config configs/thuman2.txt

test command:

modify dataset config at pointhuman_dataset.from_config
# novel view
CUDA_VISIBLE_DEVICES=3  python run_nerf_batch.py --config configs/thuman2.txt --save_weights 0

# novel pose
CUDA_VISIBLE_DEVICES=3  python run_nerf_batch.py --config configs/thuman2.txt --save_weights 0
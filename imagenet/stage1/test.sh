# export PATH=/ssd1/vis/yaomingde/miniconda3_cuda9/bin:$=PATH

# # Concat
# cd /ssd1/vis/yaomingde/VDN/code/single/N2V_CBSD68; CUDA_VISIBLE_DEVICES=7 python test.py --valid_dir /ssd1/vis/yaomingde/VDN/data/DAVIS/valid --output_path ./out_images --checkname checkpoint1800 --resume ./run/CBSD68/Unet/experiment_0/checkpoint_1800.pth.tar

# cd /ssd1/vis/yaomingde/VDN/code/single/N2V_CBSD68; CUDA_VISIBLE_DEVICES=7 python test.py --valid_dir /ssd1/vis/yaomingde/VDN/data/DAVIS/valid --output_path ./out_images --checkname checkpoint1400 --resume ./run/CBSD68/Unet/experiment_0/checkpoint_1400.pth.tar

# cd /ssd1/vis/yaomingde/VDN/code/single/N2V_CBSD68; CUDA_VISIBLE_DEVICES=7 python test.py --valid_dir /ssd1/vis/yaomingde/VDN/data/DAVIS/valid --output_path ./out_images --checkname checkpoint800 --resume ./run/CBSD68/Unet/experiment_0/checkpoint_800.pth.tar
CUDA_VISIBLE_DEVICES=7 python test.py --valid_dir /ssd1/vis/yaomingde/VDN/data/DAVIS/valid --output_path ./out_images --checkname N2V_v2_1_supcheckpoint_3800 --resume N2V_v2_1_supcheckpoint_3800.pth.tar

CUDA_VISIBLE_DEVICES=7 python test.py --valid_dir /ssd1/vis/yaomingde/VDN/data/DAVIS/valid --output_path ./out_images --checkname N2V_cut_v2_1_3checkpoint_1800 --resume N2V_cut_v2_1_3checkpoint_1800.pth.tar
CUDA_VISIBLE_DEVICES=7 python test.py --valid_dir /ssd1/vis/yaomingde/VDN/data/DAVIS/valid --output_path ./out_images --checkname N2V_cut_v2_1_3checkpoint_3800 --resume N2V_cut_v2_1_3checkpoint_3800.pth.tar

CUDA_VISIBLE_DEVICES=7 python test.py --valid_dir /ssd1/vis/yaomingde/VDN/data/DAVIS/valid --output_path ./out_images --checkname N2V_cut_v2_1_2checkpoint_1800 --resume N2V_cut_v2_1_2checkpoint_1800.pth.tar
CUDA_VISIBLE_DEVICES=7 python test.py --valid_dir /ssd1/vis/yaomingde/VDN/data/DAVIS/valid --output_path ./out_images --checkname N2V_cut_v2_1_2checkpoint_3800 --resume N2V_cut_v2_1_2checkpoint_3800.pth.tar

CUDA_VISIBLE_DEVICES=7 python test.py --valid_dir /ssd1/vis/yaomingde/VDN/data/DAVIS/valid --output_path ./out_images --checkname N2V_cut_v2_1_1checkpoint_1800 --resume N2V_cut_v2_1_1checkpoint_1800.pth.tar
CUDA_VISIBLE_DEVICES=7 python test.py --valid_dir /ssd1/vis/yaomingde/VDN/data/DAVIS/valid --output_path ./out_images --checkname N2V_cut_v2_1_1checkpoint_3800 --resume N2V_cut_v2_1_1checkpoint_3800.pth.tar
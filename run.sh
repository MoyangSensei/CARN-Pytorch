# ours  dcarn
python main.py --dir_data /home/jfy/project/data/WSI/medium_1000_200 --model CARN --data_range 0001-1000/1001-1200 --scale 2 --save testcode --n_resgroups 8 --reduction 8 --n_feats 64 --patch_size 64 --lr 0.0001 --batch_size 16 --res_scale 0.1 --seed 11 --epochs 2 --loss 0.8*L1+0.2*MSE --reset

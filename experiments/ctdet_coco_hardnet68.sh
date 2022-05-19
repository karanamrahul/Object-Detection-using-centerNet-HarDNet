cd src
# train
python main.py ctdet --exp_id coco_h68 --arch hardnet_68 --batch_size 24 --master_batch 11 --lr 5e-3 --gpus 0,1 --num_workers 16 --num_epochs 150 --lr_step 100,130
# test
python test.py ctdet --exp_id coco_h68 --arch hardnet_68 --keep_res --resume
# flip test
python test.py ctdet --exp_id coco_h68 --arch hardnet_68 --keep_res --resume --flip_test
cd ..

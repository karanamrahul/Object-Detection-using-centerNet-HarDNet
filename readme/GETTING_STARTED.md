# Getting Started

This document provides tutorials to train and evaluate CenterNet. Before getting started, make sure you have finished [installation](INSTALL.md) and [dataset setup](DATA.md).


### COCO

To evaluate COCO object detection with HarDNet-85
run

~~~
python test.py ctdet --exp_id coco_h85 --arch hardnet_85 --load_model centernet_hardnet85_coco.pth
~~~

This will give an AP of `44.0` if setup correctly. The input images are resized to `512 x 512`. You can add `--flip_test` and `--flip_test --test_scales 0.5,0.75,1,1.25,1.5` to the above commend, for flip test and multi_scale test, respectively. 


## Training

We have packed all the training scripts in the [experiments](../experiments) folder.

~~~
python main.py ctdet --exp_id coco_h85 --arch hardnet_85 --batch_size 48 --master_batch 24 --lr 1e-2 --gpus 0,1 --num_workers 16 --num_epochs 300 --lr_step 230,280
~~~


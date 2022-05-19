# Objects as Points + HarDNet
Object detection using center point detection:
![](readme/fig2.png)
> [**Objects as Points**](http://arxiv.org/abs/1904.07850)

> [**HarDNet: A Low Memory Traffic Network**](https://arxiv.org/abs/1909.00948)



## Highlights

- **Simple Algorithm:** Object as a point is a simple and elegant approach for object detections, it models an object as a single point -- the center point of its bounding box.

- **Simple Network:** A U-shape HarDNet-85 with Conv3x3, ReLU, bilinear interpolation upsampling, and Sum-to-1 layer normalization comprise the whole network. There is NO dilation/deformable convolution, nor any novel activation function being used.

- **Efficient:** CenterNet-HarDNet85 model achieves **44.3** COCO mAP (test-dev) while running at **45** FPS on an NVIDIA GTX-1080Ti GPU.

- **State of The Art:** CenterNet-HarDNet85's is faster than YOLOv4, SpineNet-49, and EfficientDet-D2


## Main results

### Object Detection on COCO validation

| Backbone     | #Param | GFLOPs | Train<br>Size | Input Size |  mAP<br>(val)  | mAP<br>(test-dev) | FPS<br>(1080ti) | Model |
| :----------: | :----: | :----: | :-----------: | :--------: | :------------: | :---------------: |:--------------: | :---: |
| HarDNet85    | 37.2M  | 123.9  |  608  |  608x608   | 44.9 | 45.3 | 32  | [Download](https://ping-chao.com/hardnet/centernet_hardnet85_coco_608.pth) |
| HarDNet85    | 37.2M  |  87.9  |  512  |  512x512   | 44.3 | 44.3 | 45  | [Download](https://ping-chao.com/hardnet/centernet_hardnet85_coco.pth) |
| HarDNet85    | 37.2M  |  58.0  |  512  |  416x416   | 42.4 | 42.4 | 53  | as above |

The model was trained with Pytorch 1.5.0 on two V100-32GB GPU for **300 epochs**(eight days). Please see [experiment](experiments/ctdet_coco_hardnet85_2x.sh) for detailed hyperperameters. Using more GPUs may require sync-batchNorm to maintain the accuracy, and the learning rate may also need to adjust. You can also check if your training/val loss is roughly aligned with our [log](experiments/ctdet_coco_hardnet85_2x.log)

HarDNet-85 results (no flipping) on COCO **test-dev2017**:
```
 //512x512:
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.443
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.629
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.483
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.235
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.476
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.590
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.355
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.579
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.612
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.381
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.660
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.805

 //608x608:
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.453
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.638
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.495
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.255
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.485
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.588
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.359
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.591
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.625
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.402
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.669
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.804
```


### Comparison with other state-of-the-art works

|     Method   |  mAP(test-dev)  | FPS @ GPU |  Training epochs |
| :----------: | :-------------: | :-------: | :--------------: |
| CenterNet-HarDNet85 |  44.3    |  45 @ 1080Ti | 300 |
|  [YOLOv4](https://github.com/pjreddie/darknet)   |  43.5    |  33 @ P100 | 300 |
|  [SpineNet-49](https://github.com/tensorflow/tpu/blob/master/models/official/detection/MODEL_ZOO.md)  |  42.8    |  42 @ V100 | 350 |
| [EfficientDet-D2](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) |  43.0 | 26.5 @ 2080Ti | 500 |

## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

## Use CenterNet

For object detection on images/ video, run:

~~~
python demo.py ctdet --demo /path/to/image/or/folder/or/video --arch hardnet_85 --load_model centernet_hardnet85_coco.pth
~~~
We provide example images in `CenterNet_ROOT/images/` (from [Detectron](https://github.com/facebookresearch/Detectron/tree/master/demo)). If set up correctly, the output should look like

<p align="center"> <img src='readme/det1.png' align="center" height="230px"> <img src='readme/det2.png' align="center" height="230px"> </p>

For webcam demo, run     

~~~
python demo.py ctdet --demo webcam --arch hardnet_85 --load_model centernet_hardnet85_coco.pth
~~~

## Real-time Demo on NVIDIA Jetson nano and AGX Xavier

| Train Size |  Input Size  |  COCO <br>AP(val) |  AP-s  |  AP-m  |  AP-L  | FP16 TRT model:<br> nano (Latency) | FP16 TRT model:<br> Xavier (Latency)|
| :------:   | :----------: |  :-------------:  | :----: | :----: | :----: | :----: | :----: |
| 512x512    |   512x512    |    43.5   |  24.5  | 47.6   |  59.4  | - | [Download](https://ping-chao.com/hardnet/ctdet_hardnet_85_512x512_xavier.trt) (49 ms) | 
| 512x512    |   416x416    |    41.5   |  20.2  | 45.1   |  59.7  | [Download](https://ping-chao.com/hardnet/ctdet_hardnet_85_416x416_nano.trt) (342 ms) | [Download](https://ping-chao.com/hardnet/ctdet_hardnet_85_416x416_xavier.trt) (37 ms) |
| 512x512    |   416x320    |    39.5   |  17.9  | 42.7   |  59.4  | [Download](https://ping-chao.com/hardnet/ctdet_hardnet_85_416x320_nano.trt) (261 ms) | [Download](https://ping-chao.com/hardnet/ctdet_hardnet_85_416x320_xavier.trt) (31 ms) |
| 512x512    |   320x320    |    37.3   |  15.1  | 40.4   |  58.4  | [Download](https://ping-chao.com/hardnet/ctdet_hardnet_85_320x320_nano.trt) (210 ms) | [Download](https://ping-chao.com/hardnet/ctdet_hardnet_85_320x320_xavier.trt) (25 ms) |
| 512x512    |   256x256    |    33.0   |  11.3  | 34.4   |  56.8  | [Download](https://ping-chao.com/hardnet/ctdet_hardnet_85_256x256_nano.trt) (117 ms) | [Download](https://ping-chao.com/hardnet/ctdet_hardnet_85_256x256_xavier.trt) (17 ms) |
| 512x512    |   224x224    |    30.1   |   8.9  | 30.4   |  54.0  | [Download](https://ping-chao.com/hardnet/ctdet_hardnet_85_224x224_nano.trt) (105 ms) | [Download](https://ping-chao.com/hardnet/ctdet_hardnet_85_224x224_xavier.trt) (16 ms) |

- Above models are converted from previous 43.6 mAP model (test-dev). For the latest 44.3 mAP model, please convert it from pytorch model
- Install NVIDIA JetPack 4.4 (TensorRT 7.1)
- Install Pytorch > 1.3 for onnx opset 11 and pycuda
- Run following commands with or without the above trt models. It will convert the pytorch model into onnx and TRT model when loading model with --load_model.
- For Jetson nano, please increase swap size to avoid freeze when building your own engines on the target (See [instructions](https://forums.developer.nvidia.com/t/creating-a-swap-file/65385))
~~~

# Demo
python demo_trt.py ctdet --demo webcam --arch hardnet_85 --load_trt ctdet_hardnet_85_416x320_xavier.trt --input_w 416 --input_h 320

# or run with any size (divided by 32) by converting a new trt model:
python demo_trt.py ctdet --demo webcam --arch hardnet_85 --load_model centernet_hardnet85_coco.pth --input_w 480 --input_h 480

# You can also run test on COCO val set with trt model, which will get ~43.2 mAP for FP16 mode:
python test_trt.py ctdet --arch hardnet_85 --load_trt ctdet_hardnet_85_512x512_xavier.trt
~~~

## Benchmark Evaluation and Training

After [installation](readme/INSTALL.md), follow the instructions in [DATA.md](readme/DATA.md) to setup the datasets. Then check [GETTING_STARTED.md](readme/GETTING_STARTED.md) to reproduce the results in the paper.
We provide scripts for all the experiments in the [experiments](experiments) folder.


## License, and Other information

Please see [original CenterNet repo](https://github.com/xingyizhou/CenterNet)
  

## Citation

For CenterNet Citation, please see [original CenterNet repo](https://github.com/xingyizhou/CenterNet)

For HarDNet Citation, please see [HarDNet repo](https://github.com/PingoLH/Pytorch-HarDNet)

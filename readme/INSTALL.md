# Installation


The code was tested on Ubuntu 16.04, with [Anaconda](https://www.anaconda.com/download) Python 3.6 and [PyTorch]((http://pytorch.org/)) v1.5.0. NVIDIA GPUs are needed for both training and testing.
After install Anaconda:


1. Install pytorch1.5.0:

     
2. Install [COCOAPI](https://github.com/cocodataset/cocoapi):

    ~~~
    # COCOAPI=/path/to/clone/cocoapi
    git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
    cd $COCOAPI/PythonAPI
    make
    python setup.py install --user
    ~~~

3. Install the requirements and clone this repo

    ~~~
    pip install -r requirements.txt
    git clone --recursive https://github.com/PingoLH/CenterNet-HarDNet.git
    ~~~   
    
4. [Optional, only required if you are using extremenet or multi-scale testing] Compile NMS if your want to use multi-scale testing or test ExtremeNet.

    ~~~
    cd $CenterNet_ROOT/src/lib/external
    make
    ~~~


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
from time import perf_counter

import torch
import numpy as np
from torch import nn
import torch.onnx

import tensorrt as trt
import pycuda.autoinit
import pycuda
import pycuda.driver as cuda

from opts import opts
from detectors.detector_factory import detector_factory
from pytorch_bn_fusion.bn_fusion import fuse_bn_recursively
from models.networks.hardnet import get_pose_net as get_hardnet
from models.model import load_model
from models.decode import ctdet_decode
from utils.debugger import Debugger
from utils.image import transform_preds

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

TRT_LOGGER = trt.Logger()  # This logger is required to build an engine


def get_img_np_nchw(filename):
    image = cv2.imread(filename)
    image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_cv = cv2.resize(image_cv, (224, 224))
    miu = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = np.array(image_cv, dtype=float) / 255.
    r = (img_np[:, :, 0] - miu[0]) / std[0]
    g = (img_np[:, :, 1] - miu[1]) / std[1]
    b = (img_np[:, :, 2] - miu[2]) / std[2]
    img_np_t = np.array([r, g, b])
    img_np_nchw = np.expand_dims(img_np_t, axis=0)
    return img_np_nchw

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def get_engine(max_batch_size=1, onnx_file_path="", engine_file_path="", \
               fp16_mode=False, int8_mode=False, overwrite=False
               ):

    def build_engine(max_batch_size):

        explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(explicit_batch) as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:

            builder.max_workspace_size = 1 << 30  # Your workspace size
            builder.max_batch_size = max_batch_size
            builder.fp16_mode = fp16_mode  # Default: False
            builder.int8_mode = int8_mode  # Default: False
            
            if int8_mode:
                raise NotImplementedError

            # Parse model file
            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))

            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                res = parser.parse(model.read())
            if res:
              print('Completed parsing of ONNX file')
              print('# Layers = ', network.num_layers)
              print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            else:
              print('Parse Failed, Layers = ', network.num_layers)
              exit()              

            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")

            if engine_file_path:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path) and not overwrite:
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    elif not os.path.exists(onnx_file_path):
        print('Cannot find any ONNX file or TRT file')
        exit()
    else:
        return build_engine(max_batch_size)


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs

class CenterHarDNet(nn.Module):
  def __init__(self, num_layers, opt):
    super().__init__()
    heads = {'hm': 80,'wh': 4}
    model = get_hardnet(num_layers=num_layers, heads=opt.heads, head_conv=opt.head_conv, trt=True)
    if opt.load_model:
      model = load_model(model, opt.load_model)
    model = fuse_bn_recursively(model)
    model.v2_transform()
    self.model = model
    mean = np.array(opt.mean, dtype=np.float32).reshape( 1, 3, 1, 1)
    std = np.array(opt.std, dtype=np.float32).reshape( 1, 3, 1, 1)
    self.mean = nn.Parameter(torch.from_numpy(mean))
    self.std = nn.Parameter(torch.from_numpy(std))
    self.max_per_image = 100
    self.num_classes = opt.num_classes
    self.scales = opt.test_scales
    self.opt = opt
    
  def forward(self, x):
    x = ( x/(torch.ones(x.shape, dtype=x.dtype, device=x.device)*255) - self.mean) / self.std
    out = self.model(x)

    hm = torch.sigmoid(out[0])
    wh = out[1]
    reg = wh[:,2:,:,:]
    wh  = wh[:,:2,:,:]
    dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K, trt=True)
    
    return dets

    
def show_det(dets, image, det_size, debugger, opt, pause=False, name=None):
  dets = dets.reshape(1, -1, dets.shape[2])
  h,w = image.shape[0:2]
  debugger.add_img(image, img_id='ctdet')
  
  c = np.array([w / 2, h / 2], dtype=np.float32)
  s = np.array([w, h], dtype=np.float32)
  dets[0, :,  :2] = transform_preds( dets[0, :, 0:2], c, s, det_size)
  dets[0, :, 2:4] = transform_preds( dets[0, :, 2:4], c, s, det_size)
  classes = dets[0, :, -1]
  
  for j in range(opt.num_classes):
    inds = (classes == j)
    top_preds = dets[0, inds, :5].astype(np.float32)
    for bbox in top_preds:
      if bbox[4] > opt.vis_thresh:
        debugger.add_coco_bbox(bbox[:4], j , bbox[4], img_id='ctdet')
        
  if name:
    print('detecting:', name)
    debugger.save_all_imgs( path='./', prefix=name)
  else:
    debugger.show_all_imgs(pause=pause)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  debugger = Debugger(dataset=opt.dataset, ipynb=(opt.debug==3),
                        theme=opt.debugger_theme)

  model = CenterHarDNet(85, opt).cuda()
  model.eval()
  
  image_size = (opt.input_w, opt.input_h)
  det_size = [image_size[0]//opt.down_ratio, image_size[1]//opt.down_ratio]
  
  onnx_model_path = "ctdet_%s_%dx%d.onnx"%(opt.arch,image_size[0], image_size[1])
  trt_engine_path = "ctdet_%s_%dx%d.trt"%(opt.arch,image_size[0], image_size[1])
  
  x = torch.randn((1, 3, image_size[1], image_size[0])).float().cuda()
  
  if opt.load_trt:
    trt_engine_path = opt.load_trt
    engine = get_engine(1, "", trt_engine_path, fp16_mode=True)
  else:
    if not opt.load_model:
      print('Please load model with --load_model')
      exit()
    print('\nStep 1: Converting ONNX... (PyTorch>1.3 is required)')
    torch.onnx.export(model,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      onnx_model_path,           # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes=None)
    print('\nStep 2: Converting TensorRT... ')
    engine = get_engine(1, onnx_model_path, trt_engine_path, fp16_mode=True, overwrite=True)
    
  outs = model(x)
  out_shape = outs.shape
  
  context = engine.create_execution_context()
  inputs, outputs, bindings, stream = allocate_buffers(engine)
  
  if not opt.demo:
    exit() 
                                
  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_FPS, 30)
    
    while True:
        t = perf_counter()
        _, img = cam.read()
        img_x = cv2.resize(img, image_size)
        x = img_x.transpose(2, 0, 1).reshape(1, 3, image_size[1], image_size[0]).astype(np.float32)
        inputs[0].host = x.reshape(-1)

        t2 = perf_counter()
        outs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        dets = outs[0].reshape(out_shape)
        t3 = perf_counter()

        show_det(dets, img, det_size, debugger, opt, False)

        print(' Latency = %.2f ms  (net=%.2f ms)'%((perf_counter()-t)*1000, (t3-t2)*1000))
        if cv2.waitKey(1) == 27:
            return  # esc to quit
  else:
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]
    
    for (image_name) in image_names:
      img = cv2.imread(image_name)
      img_x = cv2.resize(img, image_size)

      x = img_x.transpose(2, 0, 1).reshape(1, 3, image_size[1], image_size[0]).astype(np.float32)
      inputs[0].host = x.reshape(-1)

      outs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
      
      dets = outs[0].reshape(out_shape)
      show_det(dets, img, det_size, debugger, opt, True, os.path.basename(image_name))
      
if __name__ == '__main__':
  opt = opts().init()
  demo(opt)

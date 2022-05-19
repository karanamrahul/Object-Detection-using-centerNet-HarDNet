from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar

from time import perf_counter
import time
import tensorrt as trt
import pycuda.autoinit
import pycuda
import pycuda.driver as cuda

from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory
from utils.post_process import ctdet_post_process

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


def post_process(dets, meta, scale=1):
    #dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], opt.num_classes)
    for j in range(1, opt.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
      dets[0][j][:, :4] /= scale
    return dets[0]


def test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  
  out_shape = (1,100,6)
  TRT_LOGGER = trt.Logger()
  if not opt.load_trt:
    print('Please load TensorRT model with --load_trt')
    exit() 
  trt_engine_path = opt.load_trt
  with open(trt_engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
  context = engine.create_execution_context()
  inputs, outputs, bindings, stream = allocate_buffers(engine)
  
  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  
  results = {}
  times = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'pre','net', 'post']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind in range(num_iters):
    img_id = dataset.images[ind]
    img_info = dataset.coco.loadImgs(ids=[img_id])[0]
    img_path = os.path.join(dataset.img_dir, img_info['file_name'])

    t1 = time.time()
    img = cv2.imread(img_path)
    height, width = img.shape[0:2]
    c = np.array([width / 2., height / 2.], dtype=np.float32)
    s = np.array([width, height], dtype=np.float32)
    meta = {'c': c, 's': s,
            'out_height': opt.input_h // opt.down_ratio,
            'out_width':  opt.input_w // opt.down_ratio}
    
    image_size=(opt.input_w, opt.input_h)
    img_x = cv2.resize(img, image_size)
    x = img_x.transpose(2, 0, 1).reshape(1, 3, image_size[1], image_size[0]).astype(np.float32)
    inputs[0].host = x.reshape(-1)

    t2 = time.time()
    
    ### TensorRT Model Inference ###
    outs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    dets = outs[0].reshape(out_shape)
    t3 = time.time()
    dets = post_process(dets, meta, 1.0)
    t4 = time.time()
    
    times['tot']  = (t4-t1)
    times['pre']  = (t2-t1)
    times['net']  = (t3-t2)
    times['post'] = (t4-t3)
    
    results[img_id] = dets

    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(times[t])
      Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
        t, tm = avg_time_stats[t])
    bar.next()
  bar.finish()
  dataset.run_eval(results, opt.save_dir)

if __name__ == '__main__':
  opt = opts().parse()
  test(opt)
  
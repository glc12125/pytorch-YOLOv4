import sys
import os
import time
import argparse
import glob
import numpy as np
import cv2
# from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from tool.utils import *

try:
    # Sometimes python2 does not understand FileNotFoundError
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

def GiB(val):
    return val * 1 << 30

def find_sample_data(description="Runs a TensorRT Python sample", subfolder="", find_files=[]):
    '''
    Parses sample arguments.
    Args:
        description (str): Description of the sample.
        subfolder (str): The subfolder containing data relevant to this sample
        find_files (str): A list of filenames to find. Each filename will be replaced with an absolute path.
    Returns:
        str: Path of data directory.
    Raises:
        FileNotFoundError
    '''

    # Standard command-line arguments for all samples.
    kDEFAULT_DATA_ROOT = os.path.join(os.sep, "usr", "src", "tensorrt", "data")
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--datadir", help="Location of the TensorRT sample data directory.", default=kDEFAULT_DATA_ROOT)
    args, unknown_args = parser.parse_known_args()

    # If data directory is not specified, use the default.
    data_root = args.datadir
    # If the subfolder exists, append it to the path, otherwise use the provided path as-is.
    subfolder_path = os.path.join(data_root, subfolder)
    data_path = subfolder_path
    if not os.path.exists(subfolder_path):
        print("WARNING: " + subfolder_path + " does not exist. Trying " + data_root + " instead.")
        data_path = data_root

    # Make sure data directory exists.
    if not (os.path.exists(data_path)):
        raise FileNotFoundError(data_path + " does not exist. Please provide the correct data path with the -d option.")

    # Find all requested files.
    for index, f in enumerate(find_files):
        find_files[index] = os.path.abspath(os.path.join(data_path, f))
        if not os.path.exists(find_files[index]):
            raise FileNotFoundError(find_files[index] + " does not exist. Please provide the correct data path with the -d option.")

    return data_path, find_files

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine, batch_size):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:

        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dims = engine.get_binding_shape(binding)
        
        # in case batch dimension is -1 (dynamic)
        if dims[0] < 0:
            size *= -1
        
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

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


TRT_LOGGER = trt.Logger()
img_exts = ['png', 'jpg', 'JPG', 'JPEG', 'jpeg', 'PNG']
video_exts = ['MP4', 'mp4']

def main(engine_path, images_path, image_size, namesfile, show_conf):
    img_files = []
    [img_files.extend(glob.glob(images_path + '/*.' + e, recursive = False)) for e in img_exts]
    if len(img_files) == 0:
        print("There are no images in folder {}.".format(images_path))
    video_files = []
    [video_files.extend(glob.glob(images_path + '/*.' + e, recursive = False)) for e in video_exts]
    if len(video_files) == 0:
        print("There are no videos in folder {}.".format(images_path))
    if len(img_files) == 0 and len(video_files) == 0:
        print("No data to process, program will exit")
        exit(0)

    with get_engine(engine_path) as engine, engine.create_execution_context() as context:
        buffers = allocate_buffers(engine, 1)
        IN_IMAGE_H, IN_IMAGE_W = image_size
        context.set_binding_shape(0, (1, 3, IN_IMAGE_H, IN_IMAGE_W))
        class_names = load_class_names(namesfile)
        num_classes = len(class_names)
        for image_path in img_files:
            image_src = cv2.imread(image_path)
            boxes = detect(context, buffers, image_src, image_size, num_classes)
            image_base_name = image_path.split('/')[-1].split('.')[0]
            plot_boxes_cv2(image_src, boxes[0], savename='predictions_trt/' + image_base_name +'.jpg', class_names=class_names, show_confidence=show_conf)

        for video_path in video_files:
            vidcap = cv2.VideoCapture(video_path)
            if not vidcap.isOpened():
                print("Error opening {}".format(video_path))
                exit(-1)
            width  = int(vidcap.get(3))
            height  = int(vidcap.get(4))
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            video_base_name = video_path.split('/')[-1].split('.')[0]
            output_name = 'predictions_trt/' + video_base_name + '_processed.mp4'
            output_video = cv2.VideoWriter(output_name, fourcc, 20.0, (width,height))
            success, image_src = vidcap.read()
            while success:
                boxes = detect(context, buffers, image_src, image_size, num_classes)
                image_base_name = image_path.split('/')[-1].split('.')[0]
                img_with_detection = plot_boxes_cv2(image_src, boxes[0], class_names=class_names, show_confidence=show_conf)
                output_video.write(img_with_detection)
                success, image_src = vidcap.read()
            vidcap.release()
            output_video.release()

def get_engine(engine_path):
    # If a serialized engine exists, use it instead of building an engine.
    print("Reading engine from file {}".format(engine_path))
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())



def detect(context, buffers, image_src, image_size, num_classes):
    IN_IMAGE_H, IN_IMAGE_W = image_size

    ta = time.time()
    # Input
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    img_in = np.ascontiguousarray(img_in)
    print("Shape of the network input: ", img_in.shape)
    # print(img_in)

    inputs, outputs, bindings, stream = buffers
    print('Length of inputs: ', len(inputs))
    inputs[0].host = img_in

    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    print('Len of outputs: ', len(trt_outputs))

    trt_outputs[0] = trt_outputs[0].reshape(1, -1, 1, 4)
    trt_outputs[1] = trt_outputs[1].reshape(1, -1, num_classes)

    tb = time.time()

    print('-----------------------------------')
    print('    TRT inference time: %f' % (tb - ta))
    print('-----------------------------------')

    boxes = post_processing(img_in, 0.7, 0.4, trt_outputs)

    return boxes

def check_bool(show_conf_str):
    return show_conf_str.lower() in ['true', '1', 't', 'y', 'yes']

if __name__ == '__main__':
    engine_path = sys.argv[1]
    images_path = sys.argv[2]
    names_file = sys.argv[3]
    show_conf = check_bool(sys.argv[4])
    print("show_conf: {}".format(show_conf))
    if len(sys.argv) < 6:
        image_size = (416, 416)
    elif len(sys.argv) < 7:
        image_size = (int(sys.argv[5]), int(sys.argv[5]))
    else:
        image_size = (int(sys.argv[5]), int(sys.argv[6]))
    
    main(engine_path, images_path, image_size, names_file, show_conf)

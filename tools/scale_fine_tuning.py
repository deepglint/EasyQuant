import argparse
import pickle
# import caffe
# import caffe.proto.caffe_pb2 as caffe_pb2
import cv2
import numpy as np
import ncnn
import re
import shutil
import scipy.spatial.distance as dis
import sys
from collections import OrderedDict
import numpy as np
from functools import reduce
import os
import multiprocessing
import copy
import datetime



###############  end ###################################
def parse_args():
    parser = argparse.ArgumentParser(
        description='find the pretrained caffe models int8 quantize scale value')
    parser.add_argument('--proto', dest='proto',
                        help="path to deploy prototxt.", type=str)

    parser.add_argument('--model', dest='model',
                        help='path to pretrained weights', type=str)

    parser.add_argument('--param', dest='param',
                        help="path to deploy prototxt.", type=str)

    parser.add_argument('--bin', dest='bin',
                        help='path to pretrained weights', type=str)
    parser.add_argument('--table', dest='table', help='path to scale table', type=str)

    parser.add_argument('--layerDims', dest='layerDims', help='file stored layerdims', type=str)

    parser.add_argument('--search-log', dest='search_log', default='log.log',
                        help='log of search', type=str)
    args = parser.parse_args()
    return args, parser

def layerToOurputName():
    namePat = re.compile(r'\s+?name:\s+?"(.*)"')
    topPat =  re.compile(r'\s+?top:\s+?"(.*)"')
    res = {}
    with open(args.proto) as file:
        name = None
        top = None
        for line in file.readlines():
            if re.match(namePat, line):
                name = re.match(namePat, line).group(1)
            if re.match(topPat, line):
                top = re.match(topPat, line).group(1)
                res[name] = top
    return res





# def findEachLayerDim(caffe_model, net_file):
#
#     res = OrderedDict()
#     with open(net_file, 'r') as fin:
#         with open('temp.prototxt', 'w') as fout:
#             for line in fin.readlines():
#                 fout.write(line.replace('ReLU6', 'ReLU'))
#
#     net = caffe.Net('temp.prototxt', caffe_model, caffe.TEST)
#
#     img = np.random.random((224, 224, 3))
#     img = img.transpose(2, 0, 1)
#
#     net.blobs['data'].data[...] = img
#
#     output = net.forward()
#
#     params = caffe_pb2.NetParameter()
#     with open(net_file) as f:
#         text_format.Merge(f.read(), params)
#     print(net.blobs.keys())
#     for i, layer in enumerate(params.layer):
#         net_layer_name = layer.name.replace('conv', 'bn')
#         if net_layer_name in net.blobs.keys():
#             res[layer.name] = net.blobs[net_layer_name].data[0].shape
#     os.remove('temp.prototxt')
#     return res

def inferenceThread(net_fp32, net_int8, image_net, dim, layer, relu= False, image_size=224):


    net_fp32 = ncnn.net()
    net_fp32.load_param(param)
    net_fp32.load_model(bins)
    net_fp32.setInputBlobName("data")
    if relu:
        relu_name = layer.replace('conv', 'relu')
        out_blob_name = layer2OutputName[relu_name] + '_' + relu_name
    else:
        out_blob_name = layer2OutputName[layer]
    net_fp32.setOutputBlobName(out_blob_name)

    net_int8 = ncnn.net()
    net_int8.load_param('modified.param')
    net_int8.load_model("modified.bin")
    net_int8.setInputBlobName("data")
    net_int8.setOutputBlobName(out_blob_name)

    result_fp32 = np.zeros(dim).astype(np.float32)
    result_int8 = np.zeros(dim).astype(np.float32)

    net_int8.inference(image_net, result_int8, image_size, image_size)
    net_fp32.inference(image_net, result_fp32, image_size, image_size)

    result_fp32 = result_fp32.reshape((dim[0], -1))
    result_int8 = result_int8.reshape((dim[0], -1))
    cosSimilarity_allChannel = np.zeros(dim[0])
    for i in range(dim[0]):
        cosSimilarity_allChannel[i] = cosSimilatity(result_fp32[i], result_int8[i])

    return cosSimilarity_allChannel

def inferenceThread_PerLayer(image_net, dim, layer, relu= False, image_size=224):


    net_fp32 = ncnn.net()
    net_fp32.load_param(param)
    net_fp32.load_model(bins)
    net_fp32.setInputBlobName("data")
    if relu:
        relu_name = layer.replace('conv', 'relu')
        out_blob_name = layer2OutputName[relu_name] + '_' + relu_name
    else:
        out_blob_name = layer2OutputName[layer]
    net_fp32.setOutputBlobName(out_blob_name)

    net_int8 = ncnn.net()
    net_int8.load_param('modified.param')
    net_int8.load_model("modified.bin")
    net_int8.setInputBlobName("data")
    net_int8.setOutputBlobName(out_blob_name)

    result_fp32 = np.zeros(dim).astype(np.float32)
    result_int8 = np.zeros(dim).astype(np.float32)

    net_fp32.inference(image_net, result_fp32, image_size, image_size)
    net_int8.inference(image_net, result_int8, image_size, image_size)

    result_fp32 = result_fp32.reshape(-1)
    result_int8 = result_int8.reshape(-1)


    return cosSimilatity(result_fp32, result_int8)

def image_processing(image, image_size, mean_value):
    w = image.shape[1]
    h = image.shape[0]
    m = min(w, h)
    ratio = 256.0 / m
    new_w, new_h = int(ratio * w), int(ratio * h)
    image = cv2.resize(image, (new_w, new_h))
    image = image.astype(np.float32)
    top = (new_w - image_size)//2
    left = (new_h - image_size)//2
    image = image[left:left+image_size, top:top+image_size]
    image = image.transpose(2, 0, 1)
    image[0, :, :] = image[0, :, :] - mean_value[0]
    image[1, :, :] = image[1, :, :] - mean_value[1]
    image[2, :, :] = image[2, :, :] - mean_value[2]

    return image # bgr, chw, normalized


def calcChannelSimilarity(net_fp32, net_int8, imagenames, dim, layer):
    channelSimilarity = np.zeros(dim[0])
    pool = multiprocessing.Pool(processes=len(imagenames))
    poolThreads = []
    image_size = 224
    mean_value = [103.939, 116.779, 123.63]
    for imagename in imagenames:
        image = cv2.imread(imagename)
        image = image_processing(image, image_size, mean_value)
        image = image[::-1]
        image_net = image.reshape((image_size * image_size * 3,))

        if 'fc' in layer or 'branch1' in layer or 'branch2c' in layer:
            has_relu = False
        else:
            has_relu = True
        poolThreads.append(pool.apply_async(inferenceThread, (net_fp32, net_int8, image_net, dim, layer, has_relu, image_size,)))
        # channelSimilarity += inferenceThread(net_fp32, net_int8, image_net, dim, layer, has_relu, image_size,)
    #print('Threads start...')
    pool.close()
    pool.join()

    for thread in poolThreads:
        channelSimilarity += thread.get()

    return channelSimilarity / len(imagenames)


def calcLayerSimilarity(imagenames, dim, layer):
    layerSimilarity = 0.
    pool = multiprocessing.Pool(processes=len(imagenames))
    poolThreads = []
    image_size = 224
    mean_value = [103.939, 116.779, 123.63]
    for imagename in imagenames:
        image = cv2.imread(imagename)
        image = image_processing(image, image_size, mean_value)
        image = image[::-1]
        image_net = image.reshape((image_size * image_size* 3,))
        if 'fc' in layer or 'branch1' in layer or 'branch2c' in layer:
            has_relu = False
        else:
            has_relu = True
        poolThreads.append(pool.apply_async(inferenceThread_PerLayer, (image_net, dim, layer, has_relu, image_size,)))
        # layerSimilarity += inferenceThread_PerLayer(image_net, dim, layer, has_relu, image_size,)
    #print('Threads start...')
    pool.close()
    pool.join()

    for thread in poolThreads:
        layerSimilarity += thread.get()

    return layerSimilarity / len(imagenames)

def cosSimilatity(x, y):
    if np.sum(np.abs(x)) == 0 or np.sum(np.abs(y)) == 0:
        x = np.add(x, 1e-5)
        y = np.add(y, 1e-5)
    return 1 - dis.cosine(x, y)


###################### end ##############################


def modifyTable_forLayer(origin, modified, layer, channels, scales):
    fin = open(origin, 'r')
    fout = open(modified, 'w')
    modified = False
    for line in fin.readlines():
        if len(line.split()) < 5:
            fout.write(line)
        else:
            splits = line.split()
            layer_name = splits[0][:-8]
            if layer_name == layer:
                assert len(channels) == len(scales)
                for i in range(len(channels)):
                    splits[channels[i] + 1] = str(scales[i])
                fout.write(' '.join(splits))
                fout.write('\n')
                modified = True
            else:
                fout.write(line)
    fin.close()
    fout.close()
    if not modified:
        print("Not modified for layer: " + str(layer))
        exit(0)

def modifyActivationScale(origin, modified, layer, scale):
    fin = open(origin, 'r')
    fout = open(modified, 'w')
    modified = False
    for line in fin.readlines():
        if len(line.split()) > 5 or len(line.split()) < 2:
            fout.write(line)
        else:
            splits = line.split()
            layer_name = splits[0]
            if layer_name == layer:
                splits[1] = str(scale)
                fout.write(' '.join(splits))
                fout.write('\n')
                modified = True
            else:
                fout.write(line)
    fin.close()
    fout.close()
    if not modified:
        print("Not modified for layer: " + str(layer))
        exit(0)

def weight_fine_tuning(log):
    layers = collections.OrderedDict()

    with open('original.table', 'r') as f:
        for line in f.readlines():
            if len(line.split()) < 5:
                continue
            layer_name = line.split()[0][:-8]
            layer_scale = list(map(float, line.split()[1:]))
            if not layer_name in layers:
                layers[layer_name] = layer_scale

    print("weight scale fine tuning...")
    start_time = datetime.datetime.now()
    for layer in layers.keys():
        log.write('layer: {}\n'.format(layer))
        log.flush()
        print('layer {}: searching...'.format(layer))

        os.system('caffe2ncnn ' + proto + ' ' + model + ' modified.param modified.bin 0 original.table')

        res = calcChannelSimilarity(None, None, imageNames, layerDims[layer], layer)
        problem_channels = []
        channels_scale = []
        channels_cosval = []

        for channel in range(len(res)):
            problem_channels.append(channel)
            scales = [layers[layer][channel]]
            #print(layer)
            if 'proj' in layer:
                scales.extend(np.linspace(1, layers[layer][channel] * 1., 100)[1:-1])
            else:
                scales.extend(np.linspace(1, layers[layer][channel] * 1., 100)[1:-1])
            channels_scale.append(scales)
            channels_cosval.append([res[channel]])
        if len(problem_channels) == 0:
            continue
        assert len(problem_channels) == len(channels_scale) == len(channels_cosval)
        channels_scale = np.array(channels_scale)
        for i in range(1, len(channels_scale[0])):
            need_to_modify_scale = channels_scale[:, i]
            modifyTable_forLayer('original.table', 'modified.table', layer, problem_channels, need_to_modify_scale)
            os.system('caffe2ncnn ' + proto + ' ' + model + ' modified.param modified.bin 0 modified.table')

            cadidate_res = calcChannelSimilarity(None, None, imageNames, layerDims[layer], layer)
            for j in range(len(problem_channels)):
                channels_cosval[j].append(cadidate_res[problem_channels[j]])
        best_scale = []
        best_cos = []
        for k in range(len(channels_cosval)):
            index = np.argmax(channels_cosval[k])
            best_scale.append(channels_scale[k][index])
            best_cos.append(np.max(channels_cosval[k]))
        modifyTable_forLayer('original.table', 'modified.table', layer, problem_channels, best_scale)
        log.write('fixed channels\' index is: ')
        for each in problem_channels:
            log.write(str(each) + ' ')
        log.write('\n')
        for i in range(len(problem_channels)):
            log.write('channel ' + str(problem_channels[i]) + '\n')
            for scale in channels_scale[i]:
                log.write(str(scale) + ' ')
            log.write('\n')
            for val in channels_cosval[i]:
                log.write(str(val) + ' ')
            log.write('\n')
            log.write('choose scale value is: ' + str(best_scale[i]) + '| cos_val: ' + str(best_cos[i]) + '\n')
        shutil.copyfile('modified.table', 'original.table')

    end_time = datetime.datetime.now()
    print(start_time)
    print(end_time)
    print((end_time - start_time).seconds)

def activation_fine_tuning(log):
    print('activation scale fine tuning')
    start_time = datetime.datetime.now()

    layers = collections.OrderedDict()

    with open('original.table', 'r') as f:
        for line in f.readlines():
            if len(line.split()) < 2 or len(line.split()) > 5:
                continue
            layer_name = line.split()[0]
            layer_scale = float(line.split()[1])
            if not layer_name in layers:
                layers[layer_name] = layer_scale

    for layer in layers.keys():

        log.write('layer: ' + layer + '\n')
        log.flush()
        print('layer ' + layer + ": searching...")

        os.system('caffe2ncnn ' + proto + ' ' + model + ' modified.param modified.bin 0 original.table')

        res = calcLayerSimilarity(imageNames, layerDims[layer], layer)

        layer_cosval = []

        layer_scales = [layers[layer]]
        layer_scales.extend(np.linspace(layers[layer] * 0.8, layers[layer] * 1.0, 200)[1:-1])
        layer_cosval.append(res)

        for i in range(1, len(layer_scales)):

            need_to_modify_scale = layer_scales[i]
            modifyActivationScale('original.table', 'modified.table', layer, need_to_modify_scale)
            os.system('caffe2ncnn ' + proto + ' ' + model + ' modified.param modified.bin 0 modified.table')

            cadidate_res = calcLayerSimilarity(imageNames, layerDims[layer], layer)
            layer_cosval.append(cadidate_res)

        for k in range(len(layer_cosval)):
            index = np.argmax(layer_cosval)
            best_scale = layer_scales[index]
            best_cos = (np.max(layer_cosval))
        modifyActivationScale('original.table', 'modified.table', layer, best_scale)

        log.write('\n')

        for scale in layer_scales:
            log.write(str(scale) + ' ')
        log.write('\n')
        for val in layer_cosval:
            log.write(str(val) + ' ')
        log.write('\n')
        log.write('choose scale value is: ' + str(best_scale) + '| cos_val: ' + str(best_cos) + '\n')

        shutil.copyfile('modified.table', 'original.table')

    end_time = datetime.datetime.now()
    print(start_time)
    print(end_time)
    print((end_time - start_time).seconds)




global args, parser
args, parser = parse_args()
proto = args.proto
model = args.model
param = args.param
bins = args.bin

scale_table = args.table
layerDimsFile = args.layerDims
file = open(layerDimsFile, 'rb')
layerDims = pickle.load(file)
file.close()
layer2OutputName = layerToOurputName()



beginLayerIndex = 1
endLayerIndex = 110

imageRoot = 'data/calib/'
f = open("data/list50.txt", 'r')
lines = f.readlines()
imageNames = []
for line in lines:
    imageName = str(line).strip()
    if len(imageName) < 1:
        continue
    imageNames.append(imageRoot + imageName)


shutil.copyfile(scale_table, 'original.table')
shutil.copyfile('original.table', 'modified.table')
import collections

log = open(args.search_log, 'w')
weight_fine_tuning(log)
activation_fine_tuning(log)

os.system('caffe2ncnn ' + proto + ' ' + model + ' modified.param modified.bin 0 original.table')
log.close()




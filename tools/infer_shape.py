import argparse
import pickle
import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
import cv2
import numpy as np
import shutil
from google.protobuf import text_format
import scipy.spatial.distance as dis
import sys
from collections import OrderedDict
import numpy as np
from functools import reduce
import os
import re
###############  end ###################################
def parse_args():
    parser = argparse.ArgumentParser(
        description='find the pretrained caffe models int8 quantize scale value')
    parser.add_argument('--proto', dest='proto',
                        help="path to deploy prototxt.", type=str)

    parser.add_argument('--model', dest='model',
                        help='path to pretrained weights', type=str)

    parser.add_argument('--save', dest='save',
                        help='path to saved shape pkl file', type=str, default='layerDims.pickle')

    args = parser.parse_args()
    return args, parser


global args, parser
args, parser = parse_args()
proto = args.proto
model = args.model

beginLayerIndex = 1
endLayerIndex = 110

def layerToOutputName():
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

def findEachLayerDim(caffe_model, net_file):
    layer2OutputName = layerToOutputName()
    res = OrderedDict()
    with open(net_file, 'r') as fin:
        with open('temp.prototxt', 'w') as fout:
            for line in fin.readlines():
                fout.write(line.replace('ReLU6', 'ReLU'))

    net = caffe.Net('temp.prototxt', caffe_model, caffe.TEST)

    img = np.random.random((224, 224, 3))
    img = img.transpose(2, 0, 1)

    net.blobs['data'].data[...] = img

    output = net.forward()

    params = caffe_pb2.NetParameter()
    with open(net_file) as f:
        text_format.Merge(f.read(), params)
    print(net.blobs.keys())
    for i, layer in enumerate(params.layer):
        print(layer.name)
        if layer.name in layer2OutputName.keys() and layer2OutputName[layer.name] in net.blobs.keys():
            res[layer.name] = net.blobs[layer2OutputName[layer.name]].data[0].shape
    return res


def main():
    res = findEachLayerDim(args.model, args.proto)
    for k in res:
        print(k, res[k])
    import os
    os.remove('temp.prototxt')
    with open(args.save, 'w') as file:
        pickle.dump(res, file)

if __name__ == '__main__':
    main()

# BSD 3-Clause License
#
# DeepGlint is pleased to support the open source community by making EasyQuant available.
# Copyright (C) 2020 DeepGlint. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: xiaozhang
# Created Time : 2019-05-10
# File Name: generateResult.py
# Description:
"""
import os
import cv2
import numpy as np
import ncnn
from tqdm import tqdm
import multiprocessing
import argparse
import scipy.spatial.distance as dis
import glob

# Hyper-params
parser = argparse.ArgumentParser(description='PyTorch RTPose Testining')
parser.add_argument('--param_path', default='../models/resnetFaceRec_no_bn_sq.param', type=str,
                    help='path to where load param file')
parser.add_argument('--bin_path', default='../models/resnetFaceRec_no_bn_sq.bin', type=str,
                    help='path to where load bin file')
parser.add_argument('--platform', default='ncnn', type=str,
                    help='path to where load bin file')
parser.add_argument('--num_workers', default=48, type=int,
                    help='path to where load bin file')
args = parser.parse_args()

def image_processing(image, image_size, mean_value, std=[1.0, 1.0, 1.0]):
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
    image[0, :, :] = (image[0, :, :] - mean_value[0]) / std[0] # b
    image[1, :, :] = (image[1, :, :] - mean_value[1]) / std[1] # g
    image[2, :, :] = (image[2, :, :] - mean_value[2]) / std[2] # r

    return image # bgr, chw, normalized

def threadFuncCaffe(proto, caffemodel, imgs, gpuid, image_size=224, mean_value=[104., 117., 123.], std=[1.0, 1.0, 1.0]):
    import caffe
    caffe.set_device(gpuid)
    caffe.set_mode_gpu()
    net = caffe.Net(proto, caffemodel, caffe.TEST)
    res= {}
    for idx in tqdm(range(len(imgs))):
        imagepath= imgs[idx]
        image = cv2.imread(imagepath)
        if image is None:
            continue
        image = image_processing(image, image_size, mean_value, std)

        net.blobs['data'].reshape(1, 3, image_size, image_size)
        net.blobs['data'].data[...] = np.array([image], dtype=np.float32)
        out = net.forward()
        prob = out['prob']
        prob = np.squeeze(prob)
        ind = np.argsort(-prob)[0:5]
        res[imagepath]= ind
    return res

def threadFunc(param_path,bin_path,imgs, gpuid, image_size=224, mean_value=[104., 117., 123.], std=[1., 1., 1.]):
    net = ncnn.net()
    net.load_param(param_path)
    net.load_model(bin_path)
    net.setInputBlobName("data")
    net.setOutputBlobName("prob")
    res= {}
    for idx in tqdm(range(len(imgs))):
        imagepath= imgs[idx]
        image = cv2.imread(imagepath)
        if image is None:
            continue
        image = image_processing(image, image_size, mean_value, std)

        image = image[::-1]
        image = image.reshape((image_size * image_size * 3,))
        result = np.zeros((1000,))
        result = result.astype(np.float32)
        net.inference(image,result, image_size, image_size)
        result = result.astype(np.float32)
        result = result.reshape(1000)
        ind = np.argsort(-result)[0:5]
        res[imagepath]= ind
    return res

    
def main():
    import glob
    imgs= glob.glob('data/val/*.JPEG')[:50000]
    fp32_param = args.param_path
    fp32_bin= args.bin_path
    fp32_res=[]
    image_size = 224
    mean_value = [103.939, 116.779, 123.63]
    std = [1., 1., 1.]
    
    res= []
    processNum = args.num_workers # 48
    processList=[]
    pool = multiprocessing.Pool(processes=processNum)
    for i in range(processNum):
        minIndex = int(i*len(imgs)/processNum)
        maxIndex = np.min([int((i+1)*len(imgs)/processNum),len(imgs)])
        if args.platform == 'ncnn':
            fp32_res.append(pool.apply_async(threadFunc, (fp32_param,fp32_bin,imgs[minIndex:maxIndex], i%8, image_size,  mean_value, std)))
        else:
            fp32_res.append(pool.apply_async(threadFuncCaffe, (fp32_param,fp32_bin,imgs[minIndex:maxIndex], i%4, image_size, mean_value, std)))
    
    pool.close()
    pool.join()
    
    fp32_res_dict={}
    for i in fp32_res:
        fp32_res_dict.update(i.get())
    
    import pickle
    print(len(fp32_res_dict))
    with open('res_val_int88.pkl','wb') as f:
        pickle.dump(fp32_res_dict,f)
    
    gt = np.loadtxt('val1.txt')
    f = open('res_ncnn_int88.txt','w')
    for k,v in fp32_res_dict.items():
        f.write(str(k)+' '+str(v[0])+' '+str(v[1])+' '+str(v[2])+' '+str(v[3])+' '+str(v[4])+'\n')
    f.close()
    key = list(fp32_res_dict.keys())
    key.sort()
    top5 =0
    top1 =0
    for i in range(len(fp32_res_dict)):
         l = int(os.path.basename(key[i])[:-5].split('_')[-1])-1
         #print (key[i]+'\n')
         #print (gt[l],res[key[i]])
         if gt[l] in fp32_res_dict[key[i]]:
            top5+=1.
         if gt[l] == fp32_res_dict[key[i]][0]:
            top1+=1.
    print (top5)
    print (top1)
    acc_top5 = top5/len(fp32_res_dict)
    acc_top1 = top1/len(fp32_res_dict)
    print (acc_top5)
    print (acc_top1)

if __name__ == '__main__':
    main()

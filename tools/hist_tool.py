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

import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import glob

##################################################
#quant_hist[layer][0]: max_data
#quant_hist[layer][1]: hist
#quant_hist[layer][2]: hist_edges
#quant_hist[layer][3]: threshold
#################################################

quant_hist= pickle.load(open('./histgram.pkl','rb'))
i= 0
for layer in quant_hist.keys():
    layer_name= layer
    max_data= quant_hist[layer][0]
    hist,hist_edges= quant_hist[layer][1],quant_hist[layer][2]
    hist_x= (hist_edges[0:-1]+hist_edges[1:])/2
    fig = plt.figure()
    hist_x= np.log2(hist_x+1)
    hist= np.log2(hist+1)
    plt.bar(hist_x,hist)
    plt.vlines(np.log2(quant_hist[layer][3]),0,np.max(hist),'r')
    plt.title(layer_name+'    max_data:'+str(max_data))
    plt.xlabel('log2 transform')
    plt.ylabel('log2 transform')
    #plt.show()
    fig.savefig('./hist_figures/'+layer_name+'.jpg')
    i +=1
    print(i)

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

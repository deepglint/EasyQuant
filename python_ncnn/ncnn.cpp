// BSD 3-Clause License
//
// DeepGlint is pleased to support the open source community by making EasyQuant available.
// Copyright (C) 2020 DeepGlint. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <vector>
#include <iostream>

#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "ncnn.h"

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;


Net::Net()
{
    g_blob_pool_allocator.set_size_compare_ratio(0.0f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.5f);
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 1;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;

    // github recent release of ncnn
    net.opt = opt;
    // if use rq-ncnn opt set using this api
    // ncnn::set_default_option(opt);

    ncnn::set_cpu_powersave(1);
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(opt.num_threads); 
}
Net::~Net()
{
    
}
int Net::load_param(const char * paramPath)
{
    return net.load_param(paramPath);
}

int Net::load_model(const char * modelPath)
{
    return net.load_model(modelPath);
}

void Net::setInputBlobName(string name)
{
    inputBlobNmae = name;

}
void Net::setOutputBlobName(string name)
{
    outputBlobName = name;
}

int Net::inference(object & input_object,object & output_object,int inputHeight,int inputWidth)
{

    PyArrayObject* input_data_arr = reinterpret_cast<PyArrayObject*>(input_object.ptr()); 
    float * input = static_cast<float *>(PyArray_DATA(input_data_arr)); 

    PyArrayObject* output_data_arr = reinterpret_cast<PyArrayObject*>(output_object.ptr()); 
    float * output = static_cast<float *>(PyArray_DATA(output_data_arr)); 

    ncnn::Mat in = ncnn::Mat(inputWidth,inputHeight,3,4u);

    memcpy(in.channel(2),input,sizeof(float)*inputWidth*inputHeight);
    memcpy(in.channel(1),input+inputWidth*inputHeight,sizeof(float)*inputWidth*inputHeight);
    memcpy(in.channel(0),input+2*inputWidth*inputHeight,sizeof(float)*inputWidth*inputHeight);

    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    // ex.set_num_threads(4);
    ex.input(inputBlobNmae.c_str(), in);
    ncnn::Mat out;
    ex.extract(outputBlobName.c_str(),out);

    for(int i = 0;i<out.c;i++)
    {
        memcpy(output + i*out.h*out.w,out.channel(i),sizeof(float)*out.h*out.w);
    }


    // printf("%d %d %d\n", out.w, out.h, out.c);
    // for (int ic=0;ic<out.c;ic++)
    // {
    //     float maxv=0;
    //     int maxx=0;
    //     int maxy=0;
    //     for(int ih=0;ih<out.h;ih++)
    //         for (int iw=0;iw<out.w;iw++)
    //         {
    //             float val=out.channel(ic).row(ih)[iw];
    //             if (maxv<val)
    //             {
    //                 maxv=val;
    //                 maxx=iw;
    //                 maxy=ih;
    //             }
    //         }
    //     printf("now value: %f(%d,%d)\n", maxv, maxx, maxy);
    // }
    // for(int h =0;h<out.h;h++)
    // {
    //     for(int w=0;w<out.w;w++)
    //     {
    //         for(int c = 0;c<out.c;c++)
    //         {
    //             *output++ = (float)out.channel(c).row(h)[w];
    //         }
    //     }
    // }

    return 0;

}

void extract_feature_blob_f32_debug(const char* comment, const char* layer_name, const ncnn::Mat& blob);
int Net::inference_debug_writeOutputBlob2File(object & input_object,object & output_object,int inputHeight,int inputWidth)
{

    PyArrayObject* input_data_arr = reinterpret_cast<PyArrayObject*>(input_object.ptr()); 
    float * input = static_cast<float *>(PyArray_DATA(input_data_arr)); 

    ncnn::Mat in = ncnn::Mat(inputWidth,inputHeight,3,4u);

    memcpy(in.channel(2),input,sizeof(float)*inputWidth*inputHeight);
    memcpy(in.channel(1),input+inputWidth*inputHeight,sizeof(float)*inputWidth*inputHeight);
    memcpy(in.channel(0),input+2*inputWidth*inputHeight,sizeof(float)*inputWidth*inputHeight);

    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    // ex.set_num_threads(4);


    //输出各个层的结果到文件
    for(int layer_index = 1;layer_index<184;layer_index++)
    {
        char layerName[128] = {'\0'};
        sprintf(layerName, "ConvNd_%d", layer_index);
        ex.input(inputBlobNmae.c_str(), in);
        ncnn::Mat out;
        ex.extract(layerName,out);
        extract_feature_blob_f32_debug("debug",layerName,out);
    }


    return 0;

}
/*
* Extract the blob feature map
*/
void extract_feature_blob_f32_debug(const char* comment, const char* layer_name, const ncnn::Mat& blob)
{
    char file_path_output[128] = {'\0'};
    char file_dir[128] = {'\0'};

    FILE *pFile = NULL;

    std::string name = layer_name;   
    
    sprintf(file_dir, "./output/");
    mkdir(file_dir, 0777);

    sprintf(file_path_output, "./output/%s_%s_blob_data.txt", name.c_str(), comment);

    pFile = fopen(file_path_output,"w");
    if(pFile == NULL)
    {
        printf("open file error!\n");
    }

    int channel_num = blob.c;
    
    //save top feature maps
    for(int k = 0; k < channel_num; k++)
    {
        fprintf(pFile, "blob channel %d:\n", k);

        //float *data = top_blob.data + top_blob.cstep*k;
        const float *data = blob.channel(k);
        for(int i = 0; i < blob.h; i++)
        {
            for(int j = 0; j < blob.w; j++)
            {
                fprintf(pFile, "%s%8.6f ", (data[j]<0)?"":" ", data[j]);
            }
            fprintf(pFile, "\n");
            data += blob.w;
        }
        fprintf(pFile, "\n");
    }     

    //close file
    fclose(pFile);   
    pFile = NULL;
}

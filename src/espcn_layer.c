#include "espcn_layer.h"
#include "utils.h"
#include "activations.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

#ifdef AI2
#include "xnor_layer.h"
#endif





static size_t get_workspace_size(layer l){
#ifdef CUDNN
    if(gpu_index >= 0){
        size_t most = 0;
        size_t s = 0;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.weightDesc,
                l.convDesc,
                l.dstTensorDesc,
                l.fw_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dweightDesc,
                l.bf_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
                l.weightDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dsrcTensorDesc,
                l.bd_algo,
                &s);
        if (s > most) most = s;
        return most;
    }
#endif
    // printf("%d %d %d %d \n", l.out_h, l.out_w, l.c, l.groups);
    return (size_t)l.out_h*l.out_w*l.c/l.groups*sizeof(float);
}

#ifdef GPU

#endif

espcn_layer make_espcn_layer(int batch, int h, int w, int c, int n, int groups)
{
    int i;
    espcn_layer l = {0};
    l.type = ESPCN;

    l.groups = groups;
    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.batch = batch;
    l.dontsave = 1;


    // printf("\n=========groups: %d, n: %d, size: %d, c: %d, c/groups*n*size*size: %d\n", groups, n, size, c, c/groups*n*size*size);



    // float scale = 1./sqrt(size*size*c);
    // float scale = sqrt(2./(size*size*c/l.groups));
    int scale = sqrt(c/3);
    l.scale = scale;
    //printf("convscale %f\n", scale);
    //scale = .02;
    //for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
    // for(i = 0; i < l.nweights; ++i) l.weights[i] = scale*rand_normal();
    int out_w = espcn_out_width(l);
    int out_h = espcn_out_height(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

    l.forward = forward_espcn_layer;
    l.backward = backward_espcn_layer;


#ifdef GPU
    l.forward_gpu = forward_espcn_layer;
    l.backward_gpu = backward_espcn_layer;
    // l.update_gpu = update_convolutional_layer_gpu;



#ifdef CUDNN
        // cudnnCreateTensorDescriptor(&l.normTensorDesc);
        // cudnnCreateTensorDescriptor(&l.srcTensorDesc);
        // cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        // cudnnCreateFilterDescriptor(&l.weightDesc);
        // cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
        // cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
        // cudnnCreateFilterDescriptor(&l.dweightDesc);
        // cudnnCreateConvolutionDescriptor(&l.convDesc);
        // cudnn_convolutional_setup(&l);
#endif
    // }
#endif
    l.workspace_size = get_workspace_size(l);
    // printf("l.workspace_size %d\n", l.workspace_size);

    // fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);
    fprintf(stderr, "espcn \n");
    return l;
}



/*
void test_convolutional_layer()
{
    convolutional_layer l = make_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, LEAKY, 1, 0, 0, 0);
    l.batch_normalize = 1;
    float data[] = {1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3};
    //net.input = data;
    //forward_convolutional_layer(l);
}
*/



void forward_espcn_layer(espcn_layer l, network net)
{
    int i, j;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    int m = l.n/l.groups;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *c = l.output + (i*l.groups + j)*n*m;
            float *im =  net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
            
            espcn_cpu(im, l.c/l.groups,  l.scale, l.h,  l.w, c);
        }
    }
}

void backward_espcn_layer(espcn_layer l, network net)
{
    int i, j;
    int m = l.n/l.groups;
    int k = l.out_w*l.out_h;


    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.delta + (i*l.groups + j)*m*k;
            // float *b = net.workspace;
            // b = a;

            float *imd = net.delta + (i*l.groups + j)*l.c/l.groups*l.h*l.w;


            if (net.delta) {
                
                reverse_espcn_cpu(a, l.n,  l.scale, l.out_h,  l.out_w, imd);
            }
        }
    }
}

int espcn_out_height(espcn_layer l)
{
    return l.h*l.scale;
}

int espcn_out_width(espcn_layer l)
{
    return l.w*l.scale;
}






void espcn_cpu(float* data_im,
     int channels,  int scale, int height,  int width, float* data_col)
{
    int c,h,w;
    
    for (h = 0; h < height; ++h){
        for (w = 0; w < width; ++w){
            for (c = 0; c < channels; ++c){
                // int im_row = scale*h;
                // int im_col = scale*w;
                int im_ch = c / (scale * scale);
                int im_pos =( c / scale) % scale;
                int col_index = (im_ch * scale * height + scale * h) * (scale*width) + (scale * w) + (im_pos*scale*width) + (c % scale);
                data_col[col_index] = data_im[(c * height + h) * width + w];            
            }
        }
    }
}

void reverse_espcn_cpu(float* data_im,
     int channels,  int scale, int height,  int width, float* data_col)
{
    int out_height = height / scale;
    int out_width = width / scale;
    int h,w,c;
    for (h=0; h<height; ++h){
        for (w=0; w<width; ++w){
            for (c=0; c<channels; ++c){
                int out_h = h / scale;
                int out_w = w / scale;
                int out_c = c*(scale*scale) + (h % scale) * scale + (w % scale);
                int col_index = out_width*(out_height*out_c + out_h) + out_w;
                data_col[col_index] = data_im[width*(height*c + h)+w];
            }
        }
    }
}



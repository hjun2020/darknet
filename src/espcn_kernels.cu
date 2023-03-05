#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "convolutional_layer.h"
#include "espcn_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}


__global__ void espcn_forward_gpu_kernel(const float* data_im,
        const int height_in, const int width_in, 
        const int espcn_scale,
        const int height_out, const int width_out,
        float *data_col) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;

    int w_out = index % width_out;
    int h_index = index / width_out;
    int h_out = h_index % height_out;
    int c_out = index / (width_out * height_out);

    int w_in = w_out / espcn_scale;
    int h_in = h_out / espcn_scale;
    int c_in = c_out * (espcn_scale * espcn_scale);

    const float* data_im_ptr = data_im;
    float* data_col_ptr = data_col;


    data_col_ptr[index] = data_im_ptr[c_in * width_in * height_in + width_in * h_in + w_in];

}




void forward_espcn_layer_gpu(espcn_layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);

// #ifdef CUDNN

// #else
    int i, j;
    int m = l.n/l.groups;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *c = l.output_gpu + (i*l.groups + j)*n*m;
            float *im = net.input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
            int num_kernels = l.out_w*l.out_h * 3; 
            // espcn_forward_gpu_kernel(im, l.h, l.w, l.espcn_scale, l.out_h, l.out_w, c);
            espcn_forward_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK, BLOCK>>>(im, l.h, l.w, 3,l.out_h, l.out_w,c);
        }
    }
// #endif
}

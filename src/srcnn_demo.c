#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include <sys/time.h>



#define DEMO 1

#ifdef OPENCV




static float **pred_buffer [3];
static int buff_index = 0;
static network *net;
static image out_im_buffer [3];
static void *input_mat_buffer [3];
static void *output_mat_buffer [3];
static image input_im_buffer[3];
static float **network_input_buffer [3];
static void * cap;
static int demo_done = 0;



static void *predict_in_thread_srcnn(void *ptr)
{
    memcpy(pred_buffer[(buff_index+1)%3], network_predict(net, network_input_buffer[(buff_index+1)%3]), net->batch*net->outputs*sizeof(float));
}

static void *load_input_mat_demo(void *ptr)
{
    // free_image(input_im_buffer[buff_index]);
    // input_im_buffer[buff_index%3] = load_image_color("data/scream.jpg", 0, 0);
    // input_im_buffer[buff_index%3] = get_image_from_stream(cap);

    // free_image(input_im_buffer[buff_index%3]);
    free(input_mat_buffer[buff_index%3]);
    input_mat_buffer[buff_index%3]= get_mat_from_stream(cap);
    // if(input_mat_buffer[buff_index%3].data == 0) {
    //     demo_done = 1;
    //     return 0;
    // }
    return 0;

}

static void *load_input_im_demo(void *ptr)
{
    free_image(input_im_buffer[(buff_index+1)%3]);
    input_im_buffer[(buff_index+1)%3]= get_luminance(input_mat_buffer[(buff_index+1)%3]);
    // image temp= get_luminance(input_mat_buffer[(buff_index+1)%3]);
    // save_image(temp, "data_test/bal");

}


static void *data_prep_in_thread_srcnn(void *ptr)
{
    load_args_espcn args = *(load_args_espcn *)ptr;

    int num_cols = args.num_cols;
    int num_rows = args.num_rows;
    int w_len = args.w_len;
    int h_len = args.h_len;
    int w_offset = args.w_offset;
    int h_offset = args.h_offset;
    int w_extra_offset = args.w_extra_offset;
    int h_extra_offset = args.h_extra_offset;
    int n = args.n;
    int out_c = args.out_c;
    int out_h = args.out_h;
    int out_w = args.out_w;
    int i;
    for(i = 0; i < n; ++i){
        int start_col = i % num_cols;
        int start_row = i / num_cols;
        int w_start = w_len * start_col - (w_offset * start_col);
        int h_start = h_len * start_row - (h_offset * start_row);
        if(start_col == num_cols - 1){
            w_start = w_start - w_extra_offset;
        }
        if(start_row == num_rows -1){
            h_start = h_start - h_extra_offset;
        }

        load_partial_data_demo(input_im_buffer[(buff_index+2)%3].data, i, h_start, w_start, h_len, w_len, out_c, out_h, out_w, network_input_buffer[(buff_index+2)%3]);
        
    }

}

static void *mat_to_image_in_thread_srcnn(void *ptr)
{
    out_im_buffer[(buff_index+4)%3] = mat_to_image_ptr(output_mat_buffer[(buff_index+4)%3]);
}

static void *merge_in_thread_srcnn(void *ptr)
{
    load_args_espcn args = *(load_args_espcn *)ptr;
    // free_image(out_im_buffer[(buff_index+3)%3]);
    ///////////////////temp////////////////////////////////////////////////// pred_buffer ----> network_input_buffer
    // out_im_buffer[(buff_index+3)%3] = float2im(args, pred_buffer[(buff_index+3)%3]);  
    image temp = float2im(args, pred_buffer[(buff_index+3)%3]);
    // save_image(temp, "data_test/METAL1");
    free(output_mat_buffer[(buff_index+3)%3]);
    output_mat_buffer[(buff_index+3)%3] = merge_ycbcr(input_mat_buffer[(buff_index+3)%3], temp);
    return 0;
}

void *display_in_thread_srcnn_demo(void *ptr)
{
    int c = show_image(out_im_buffer[(buff_index + 1)%3], "Demo", 1);
    // free_image(out_im_buffer[(buff_index + 1)%3]);
    // if (c != -1) c = c%256;
    // if (c == 27) {
    //     demo_done = 1;
    //     return 0;
    // } else if (c == 82) {
    //     demo_thresh += .02;
    // } else if (c == 84) {
    //     demo_thresh -= .02;
    //     if(demo_thresh <= .02) demo_thresh = .02;
    // } else if (c == 83) {
    //     demo_hier += .02;
    // } else if (c == 81) {
    //     demo_hier -= .02;
    //     if(demo_hier <= .0) demo_hier = .0;
    // }
    return 0;
}





void srcnn_video_demo(char *datacfg, char *cfgfile, char *weightfile, char *filename, int *gpus, int ngpus, int clear)
{
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.list");
    char *backup_directory = option_find_str(options, "backup", "/backup/");


    // srand(time(0));
    char *base = basecfg(cfgfile);
    float avg_loss = -1;
    network **nets = calloc(ngpus, sizeof(network));

    // srand(time(0));
    // int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network_with_size_option(cfgfile, weightfile, clear, 200, 200, 1, 60, 1);
        nets[i]->learning_rate *= ngpus;
    }
    // srand(time(0));
    net = nets[0];

    if(filename){
    printf("video file: %s\n", filename);
    cap = open_video_stream(filename, 0, 0, 0, 0);
    }

    load_args_espcn args = {0};
    data buffer;
    // image orig = load_image_color("data/scream.jpg", 0, 0);
    image orig = get_image_from_stream(cap);
    printf("%d, %d\n", orig.w, orig.h);
    args.in_c = 1;
    args.in_h = net->h;
    args.in_w = net->w;
    args.out_c = 1;
    args.out_h = orig.h*3;
    args.out_w = orig.w*3;
    args.num_rows = args.out_h / args.in_h + 1;
    args.num_cols = args.out_w / args.in_h + 1;
    args.h_offset = (args.in_h * args.num_rows - args.out_h) / (args.num_rows - 1);
    args.w_offset = (args.in_w * args.num_cols - args.out_w) / (args.num_cols - 1);
    args.h_extra_offset = (args.in_h * args.num_rows - args.out_h) % (args.num_rows - 1);
    args.w_extra_offset = (args.in_w * args.num_cols - args.out_w) % (args.num_cols - 1);

    args.espcn_scale = sqrt(net->outputs / net->inputs);

    args.in_w_pred = args.in_w * args.espcn_scale;
    args.in_h_pred = args.in_h * args.espcn_scale;
    args.in_c_pred = args.in_c;
    args.out_w_pred = args.out_w * args.espcn_scale;
    args.out_h_pred = args.out_h * args.espcn_scale;
    args.out_c_pred = args.out_c;
    args.w_offset_pred = args.w_offset * args.espcn_scale;
    args.h_offset_pred = args.h_offset * args.espcn_scale;
    args.w_extra_offset_pred = args.w_extra_offset * args.espcn_scale;
    args.h_extra_offset_pred = args.h_extra_offset * args.espcn_scale; 

    args.h_len = 200;
    args.w_len = 200;
    args.im_data = orig.data;
    args.threads = args.num_cols * args.num_rows;
    args.n = args.num_cols * args.num_rows;
    args.d = &buffer;
    args.type = ESPCN_DEMO_DATA;

    net->batch = args.n;
    net->subdivisions = 1;

    printf("%d!!!!!!!!!!!!!!! \n", args.n);


    input_mat_buffer[0]= get_mat_from_stream(cap);
    input_mat_buffer[1]= get_mat_from_stream(cap);
    input_mat_buffer[2]= get_mat_from_stream(cap);

    input_im_buffer[0] = get_luminance(input_mat_buffer[0]);
    input_im_buffer[1] = get_luminance(input_mat_buffer[1]);
    input_im_buffer[2] = get_luminance(input_mat_buffer[2]);

    pred_buffer[0] = calloc(net->outputs*args.n, sizeof(float));
    pred_buffer[1] = calloc(net->outputs*args.n, sizeof(float));
    pred_buffer[2] = calloc(net->outputs*args.n, sizeof(float));

    network_input_buffer[0] = calloc(net->inputs*args.n, sizeof(float));
    network_input_buffer[1] = calloc(net->inputs*args.n, sizeof(float));
    network_input_buffer[2] = calloc(net->inputs*args.n, sizeof(float));

    output_mat_buffer[0] = get_mat_from_stream(cap);
    output_mat_buffer[1] = get_mat_from_stream(cap);
    output_mat_buffer[2] = get_mat_from_stream(cap);


    // input_im_buffer[0] = load_image_color(filename, 0, 0);
    // input_im_buffer[1] = load_image_color(filename, 0, 0);
    // input_im_buffer[2] = load_image_color(filename, 0, 0);
    // input_im_buffer[0] = copy_image(orig);
    // input_im_buffer[1] = copy_image(orig);
    // input_im_buffer[2] = copy_image(orig);



    data data_buffer [3];    

    pthread_t predict_thread;
    pthread_t input_y_im_thread;
    pthread_t merge_thread;
    pthread_t input_thread;
    pthread_t data_pred_thread;
    pthread_t output_thread;


    printf("%d, %d, %d, %d, %d, %d %d %d\n\n", args.out_w, args.out_h, args.num_rows, args.num_cols, args.h_offset, args.w_offset, args.h_extra_offset, args.w_extra_offset);
    // printf("%d, %d, %d, %d, %d, %d\n\n", args.num_rows, args.num_cols, args.h_offset_pred, args.w_offset_pred, args.h_extra_offset_pred, args.w_extra_offset_pred);

    // pthread_t load_thread = load_data_espcn(args);

    // pthread_join(load_thread, 0);

    data d = *args.d;
    // should be fixed
    d.X.cols = args.in_w*args.in_h*args.in_c;
    //////////////

    struct load_args_espcn *ptr = calloc(1, sizeof(struct load_args_espcn));
    double time=what_time_is_it_now();
    *ptr = args;
    int count = 0;
    while(!demo_done){
        // memcpy(pred_buffer[t%3], network_predict_data_to_float(net, d), net->outputs*args.n*sizeof(float));
        if(pthread_create(&input_thread, 0, load_input_mat_demo, ptr)) error("Thread creation failed");
        if(pthread_create(&input_y_im_thread, 0, load_input_im_demo, ptr)) error("Thread creation failed");
        if(pthread_create(&data_pred_thread, 0, data_prep_in_thread_srcnn, ptr)) error("Thread creation failed");
        // if(count > 5){

            if(pthread_create(&predict_thread, 0, predict_in_thread_srcnn, ptr)) error("Thread creation failed");
        // }
        if(pthread_create(&merge_thread, 0, merge_in_thread_srcnn, ptr)) error("Thread creation failed");
        if(pthread_create(&output_thread, 0, mat_to_image_in_thread_srcnn, ptr)) error("Thread creation failed");

        if(count > 4) display_in_thread_srcnn_demo(0);

        pthread_join(input_thread,0);
        pthread_join(input_y_im_thread,0);
        pthread_join(data_pred_thread,0);
        // if(count > 5){
            pthread_join(predict_thread,0);
        // }
        pthread_join(merge_thread,0);
        pthread_join(output_thread,0);

        buff_index = (buff_index+1)%3;
        // if(t%1000 == 0) printf("count: %d\n", t);
        // if(count > 4) imshow("video", out_im_buffer[buff_index]);
        count++;

        if(count>300) break;
    }

    

    printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);
    printf("%d\n", count);
    // save_image(out_im_buffer[2], "data_test/bal");
  

    // image im = float2im(args, pred_buffer[2]);
    // save_image(out_im_buffer[1], "data_test/tt11");

    // free_image(im);

    return;
}

#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg, float hier, int w, int h, int frames, int fullscreen)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif


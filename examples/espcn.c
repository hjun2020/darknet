#include "darknet.h"
#include "stb_image.h"
#include "image.h"

static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};


void generateKernel(float *kernel, int size, float sigma) {
    float sum = 0.0f;
    int i, j;
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            int x = i - size / 2;
            int y = j - size / 2;
            kernel[i * size + j] = exp(-(x*x + y*y) / (2 * sigma*sigma));
            sum += kernel[i * size + j];
        }
    }
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            kernel[i * size + j] /= sum;
        }
    }
}

void train_enhencer(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.list");
    char *backup_directory = option_find_str(options, "backup", "/backup/");


    // printf("%s\n", train_images);

    
    srand(time(0));
    char *base = basecfg(cfgfile);
    float avg_loss = -1;
    network **nets = calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network_espcn(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    data train, buffer;

    layer l = net->layers[net->n - 1];

    int classes = l.classes;
    float jitter = l.jitter;
    


    list *plist = get_paths(train_images);

    char **paths = (char **)list_to_array(plist);

    load_args args = get_base_args(net);
    args.coords = l.coords;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = ENHENCE_DATA;

    args.espcn_scale = (int)sqrt(net->outputs / net->inputs);
    args.gaussian_filter = malloc(sizeof(float) * 7 * 7);
    generateKernel(args.gaussian_filter, 7, 1.0f);

    args.threads = 64;


    pthread_t load_thread = load_data(args);



    double time;
    int count = 0;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net->max_batches){
        if(l.random && count++%10 == 0){
            printf("Resizing\n");
            int dim = (rand() % 10 + 10) * 32;
            if (get_current_batch(net)+200 > net->max_batches) dim = 608;
            //int dim = (rand() % 4 + 16) * 32;

            // printf("%d\n", dim);

            args.w = dim;
            args.h = dim;

            pthread_join(load_thread, 0);

            train = buffer;


            free_data(train);
            load_thread = load_data(args);

            #pragma omp parallel for
            for(i = 0; i < ngpus; ++i){
                resize_network(nets[i], dim, dim);
            }
            net = nets[0];
        }
        time=what_time_is_it_now();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);


        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);


        time=what_time_is_it_now();
        float loss = 0;

#ifdef GPU
        if(ngpus == 1){
            loss = train_network_espcn(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        i = get_current_batch(net);
        printf("%ld: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, i*imgs);
        if(i%100==0){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s.backup_test", backup_directory, base);
            save_weights(net, buff);
        }
        if(i%10000==0 || (i < 1000 && i%100 == 0)){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}


static int get_coco_image_id(char *filename)
{
    char *p = strrchr(filename, '/');
    char *c = strrchr(filename, '_');
    if(c) p = c;
    return atoi(p+1);
}

static void print_cocos(FILE *fp, char *image_path, detection *dets, int num_boxes, int classes, int w, int h)
{
    int i, j;
    int image_id = get_coco_image_id(image_path);
    for(i = 0; i < num_boxes; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        float bx = xmin;
        float by = ymin;
        float bw = xmax - xmin;
        float bh = ymax - ymin;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fp, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, dets[i].prob[j]);
        }
    }
}

// void temp_test(char *cfgfile){

// }

// static float **pred_buffer [3];
// static int buff_index = 0;
// static network *net;
// static image out_im_buffer [3];
// static image input_im_buffer [3];
// static float **network_input_buffer [3];


// // void *predict_in_thread(void *ptr)
// // {
// //     load_args_espcn args = *(load_args_espcn *)ptr;
// //     data d = *args.d;
// //     // should be fixed
// //     d.X.cols = args.in_w*args.in_h*args.in_c;
// //     float *out = network_predict_data_to_float(net, *args.d);
// //     memcpy(pred_buffer[buff_index%3], out, net->outputs*args.n*sizeof(float));
    
// //     return 0;
// // }

// void *predict_in_thread(void *ptr)
// {
//     memcpy(pred_buffer[(buff_index+1)%3], network_predict(net, network_input_buffer[(buff_index+1)%3]), net->batch*net->outputs*sizeof(float));
// }


// void *merge_in_thread(void *ptr)
// {
//     load_args_espcn args = *(load_args_espcn *)ptr;
//     free_image(out_im_buffer[(buff_index+3)%3]);
//     ///////////////////temp////////////////////////////////////////////////// pred_buffer ----> network_input_buffer
//     out_im_buffer[(buff_index+3)%3] = float2im(args, pred_buffer[(buff_index+3)%3]);    
//     return 0;
// }

// void load_partial_data_demo(float *im, int n, int h_start, int w_start, int h_len, int w_len, int c, int h, int w, float *dst_buff)
// {
//     int i,j,k;
//     int start_offset = n*h_len*w_len*3;
//     for(k = 0; k < c; ++k){
//         for(j = h_start; j < h_start + h_len; ++j){
//             for(i = w_start; i < w_start + w_len; ++i){
//                 int dst_index = (i-w_start) + w_len*(j-h_start) + w_len*h_len*k;
//                 int src_index = i + w*j + k*h*w;
//                 dst_buff[start_offset+dst_index] = (float)im[src_index];
//             }
//         }
//     }

//     return 0;
// }

// void *data_prep_in_thread(void *ptr)
// {
//     load_args_espcn args = *(load_args_espcn *)ptr;

//     int num_cols = args.num_cols;
//     int num_rows = args.num_rows;
//     int w_len = args.w_len;
//     int h_len = args.h_len;
//     int w_offset = args.w_offset;
//     int h_offset = args.h_offset;
//     int w_extra_offset = args.w_extra_offset;
//     int h_extra_offset = args.h_extra_offset;
//     int n = args.n;
//     int out_c = args.out_c;
//     int out_h = args.out_h;
//     int out_w = args.out_w;
//     int i;
//     for(i = 0; i < n; ++i){
//         int start_col = i % num_cols;
//         int start_row = i / num_cols;
//         int w_start = w_len * start_col - (w_offset * start_col);
//         int h_start = h_len * start_row - (h_offset * start_row);
//         if(start_col == num_cols - 1){
//             w_start = w_start - w_extra_offset;
//         }
//         if(start_row == num_rows -1){
//             h_start = h_start - h_extra_offset;
//         }

//         load_partial_data_demo(input_im_buffer[(buff_index+2)%3].data, i, h_start, w_start, h_len, w_len, out_c, out_h, out_w, network_input_buffer[(buff_index+2)%3]);
        
//     }

// }

// void *load_input_im_demo(void *ptr)
// {
//     // free_image(input_im_buffer[buff_index]);
//     input_im_buffer[buff_index%3] = load_image_color("data/scream.jpg", 0, 0);

// }







// void data_test(char *datacfg, char *cfgfile, char *weightfile, char *filename, int *gpus, int ngpus, int clear)
// {
//     list *options = read_data_cfg(datacfg);
//     char *train_images = option_find_str(options, "train", "data/train.list");
//     char *backup_directory = option_find_str(options, "backup", "/backup/");


//     // srand(time(0));
//     char *base = basecfg(cfgfile);
//     float avg_loss = -1;
//     network **nets = calloc(ngpus, sizeof(network));

//     // srand(time(0));
//     // int seed = rand();
//     int i;
//     for(i = 0; i < ngpus; ++i){
// #ifdef GPU
//         cuda_set_device(gpus[i]);
// #endif
//         nets[i] = load_network(cfgfile, weightfile, clear);
//         nets[i]->learning_rate *= ngpus;
//     }
//     // srand(time(0));
//     net = nets[0];


//     load_args_espcn args = {0};
//     data buffer;
//     image orig = load_image_color(filename, 0, 0);
//     printf("%d, %d\n", orig.h, orig.w);
//     args.in_c = 3;
//     args.in_h = net->h;
//     args.in_w = net->w;
//     args.out_c = 3;
//     args.out_h = orig.h;
//     args.out_w = orig.w;
//     args.num_rows = args.out_h / args.in_h + 1;
//     args.num_cols = args.out_w / args.in_h + 1;
//     args.h_offset = (args.in_h * args.num_rows - args.out_h) / (args.num_rows - 1);
//     args.w_offset = (args.in_w * args.num_cols - args.out_w) / (args.num_cols - 1);
//     args.h_extra_offset = (args.in_h * args.num_rows - args.out_h) % (args.num_rows - 1);
//     args.w_extra_offset = (args.in_w * args.num_cols - args.out_w) % (args.num_cols - 1);

//     args.espcn_scale = 3;

//     args.in_w_pred = args.in_w * args.espcn_scale;
//     args.in_h_pred = args.in_h * args.espcn_scale;
//     args.in_c_pred = args.in_c;
//     args.out_w_pred = args.out_w * args.espcn_scale;
//     args.out_h_pred = args.out_h * args.espcn_scale;
//     args.out_c_pred = args.out_c;
//     args.w_offset_pred = args.w_offset * args.espcn_scale;
//     args.h_offset_pred = args.h_offset * args.espcn_scale;
//     args.w_extra_offset_pred = args.w_extra_offset * args.espcn_scale;
//     args.h_extra_offset_pred = args.h_extra_offset * args.espcn_scale; 

//     args.h_len = 104;
//     args.w_len = 104;
//     args.im_data = orig.data;
//     args.threads = args.num_cols * args.num_rows;
//     args.n = args.num_cols * args.num_rows;
//     args.d = &buffer;
//     args.type = ESPCN_DEMO_DATA;

//     net->batch = args.n;
//     net->subdivisions = 1;



//     pred_buffer[0] = calloc(net->outputs*args.n, sizeof(float));
//     pred_buffer[1] = calloc(net->outputs*args.n, sizeof(float));
//     pred_buffer[2] = calloc(net->outputs*args.n, sizeof(float));

//     network_input_buffer[0] = calloc(net->inputs*args.n, sizeof(float));
//     network_input_buffer[1] = calloc(net->inputs*args.n, sizeof(float));
//     network_input_buffer[2] = calloc(net->inputs*args.n, sizeof(float));

//     input_im_buffer[0] = load_image_color(filename, 0, 0);
//     input_im_buffer[1] = load_image_color(filename, 0, 0);
//     input_im_buffer[2] = load_image_color(filename, 0, 0);


//     data data_buffer [3];    

//     pthread_t predict_thread;
//     pthread_t merge_thread;
//     pthread_t input_thread;
//     pthread_t data_pred_thread;


//     // printf("%d, %d, %d, %d, %d, %d\n\n", args.num_rows, args.num_cols, args.h_offset, args.w_offset, args.h_extra_offset, args.w_extra_offset);
//     // printf("%d, %d, %d, %d, %d, %d\n\n", args.num_rows, args.num_cols, args.h_offset_pred, args.w_offset_pred, args.h_extra_offset_pred, args.w_extra_offset_pred);

//     pthread_t load_thread = load_data_espcn(args);

//     pthread_join(load_thread, 0);

//     data d = *args.d;
//     // should be fixed
//     d.X.cols = args.in_w*args.in_h*args.in_c;
//     //////////////

//     struct load_args_espcn *ptr = calloc(1, sizeof(struct load_args_espcn));
//     double time=what_time_is_it_now();
//     for(int t=0; t<100; t++){
//         *ptr = args;
//         // memcpy(pred_buffer[t%3], network_predict_data_to_float(net, d), net->outputs*args.n*sizeof(float));
//         if(pthread_create(&input_thread, 0, load_input_im_demo, ptr)) error("Thread creation failed");
//         if(pthread_create(&data_pred_thread, 0, data_prep_in_thread, ptr)) error("Thread creation failed");
//         if(pthread_create(&predict_thread, 0, predict_in_thread, ptr)) error("Thread creation failed");
//         if(pthread_create(&merge_thread, 0, merge_in_thread, ptr)) error("Thread creation failed");


//         pthread_join(input_thread,0);
//         pthread_join(predict_thread,0);
//         pthread_join(data_pred_thread,0);
//         pthread_join(merge_thread,0);

//         buff_index = (buff_index+1)%3;
//         if(t%1000 == 0) printf("count: %d\n", t);
//     }
//     printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);
  

//     // image im = float2im(args, pred_buffer[2]);
//     save_image(out_im_buffer[1], "data_test/tt11");

//     // free_image(im);

//     return;
// }
  

void run_enhancer(int argc, char **argv)
{   
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .5);
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    int avg = find_int_arg(argc, argv, "-avg", 3);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    int *gpus = 0;
    int gpu = 0;  
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");
    int fullscreen = find_arg(argc, argv, "-fullscreen");
    int width = find_int_arg(argc, argv, "-w", 0);  
    int height = find_int_arg(argc, argv, "-h", 0);
    int fps = find_int_arg(argc, argv, "-fps", 0);
    //int class = find_int_arg(argc, argv, "-class", 0);

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    // if(0==strcmp(argv[2], "test")) test_enhencer(datacfg, cfg, weights, filename, thresh, hier_thresh, outfile, fullscreen);
    if(0==strcmp(argv[2], "train")) train_enhencer(datacfg, cfg, weights, gpus, ngpus, clear);
    // else if(0==strcmp(argv[2], "data_test")) data_test(datacfg, cfg, weights, filename, gpus, ngpus, clear);
    else if(0==strcmp(argv[2], "espcn_video_demo")) espcn_video_demo(datacfg, cfg, weights, filename, gpus, ngpus, clear);
    // else if(0==strcmp(argv[2], "valid")) validate_enhencer(datacfg, cfg, weights, outfile);
    // else if(0==strcmp(argv[2], "valid2")) validate_enhencer_flip(datacfg, cfg, weights, outfile);
    // else if(0==strcmp(argv[2], "recall")) validate_enhencer_recall(cfg, weights);
    // else if(0==strcmp(argv[2], "demo")) {
    //     list *options = read_data_cfg(datacfg);
    //     int classes = option_find_int(options, "classes", 20);
    //     char *name_list = option_find_str(options, "names", "data/names.list");
    //     char **names = get_labels(name_list);
    //     demo(cfg, weights, thresh, cam_index, filename, names, classes, frame_skip, prefix, avg, hier_thresh, width, height, fps, fullscreen);
    // }
    //else if(0==strcmp(argv[2], "extract")) extract_detector(datacfg, cfg, weights, cam_index, filename, class, thresh, frame_skip);
    //else if(0==strcmp(argv[2], "censor")) censor_detector(datacfg, cfg, weights, cam_index, filename, class, thresh, frame_skip);
}
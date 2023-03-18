#include "darknet.h"
#include "stb_image.h"

static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};


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
        nets[i] = load_network(cfgfile, weightfile, clear);
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
            loss = train_network(net, train);
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

// images load_partial_enhenced_images_stb(char *filename, network *net, int channels, int out_w, int out_h, int in_w, int in_h)
// {
//     images images;
//     int w_remainder = in_w % out_w;
//     int h_remainder = in_h % out_h; 
//     int row;
//     int col;
//     if (w_remainder != 0){
//         col = in_w / out_w + 1;
//     } else {
//         col = in_w / out_w;
//     }
//     if (h_remainder != 0){
//         row = in_h / out_h + 1;
//     } else {
//         row = in_h / out_h;
//     }
//     col = in_w / out_w + 2;
//     row = in_h / out_h + 2;

//     int row_offset = (row * out_h - in_h) / (row-1);
//     int col_offset = (col * out_w - in_w) / (col-1);

//     int row_remainder = (row * out_h - in_h) % (row-1);
//     int col_remainder = (col * out_w - in_w) % (col-1);



//     images.data = calloc(row * col, sizeof(image));
//     images.row = row;
//     images.col = col;
    
//     int w_len = out_w;
//     int h_len = out_h;
//     images.w = 3*w_len;
//     images.h = 3*h_len;

//     for (int i=0; i<row; i++){
//         for (int j=0; j<col; j++) {
//             int r_offset = j*row_offset;
//             int c_offset = i*col_offset;
//             if(i == row-1){
//                 c_offset += col_remainder;
//             }
//             if(j == col-1){
//                 r_offset += row_remainder;
//             }
//             // images.data[i*col+j].h = out_h*3;
//             // images.data[i*col+j].w = out_w*3;
//             // images.data[i*col+j].data = network_predict(net, load_partial_image_stb(filename, 3, j*w_len-r_offset, w_len, i*h_len-c_offset, h_len).data);

//             image partial_img = load_partial_image_stb(filename, 3, j*w_len-r_offset, w_len, i*h_len-c_offset, h_len);
//             image temp_image = make_image(3*out_w, 3*out_h, 3);
//             temp_image.data = network_predict(net, partial_img.data);
//             images.data[i*col+j] = copy_image(temp_image);
//             // printf("row: %d, col: %d, w_range: %d %d, h_range: %d %d\n", i, j, j*w_len-r_offset, w_len, i*h_len-c_offset, h_len);
//         }
//     }
//     return images;
// }

void temp_test(char *cfgfile){
    network *net = parse_network_cfg(cfgfile);
    

}

void data_test(char *datacfg, char *cfgfile, char *weightfile, char *filename, int *gpus, int ngpus, int clear)
{
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.list");
    char *backup_directory = option_find_str(options, "backup", "/backup/");


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
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];
    
    load_args_espcn args = {0};
    data buffer;
    printf("test data!!!!!\n");
    image orig = load_image_color(filename, 0, 0);
    printf("%d, %d\n", orig.h, orig.w);
    args.in_c = 3;
    args.in_h = 104;
    args.in_w = 104;
    args.out_c = 3;
    args.out_h = orig.h;
    args.out_w = orig.w;
    args.num_rows = args.out_h / args.in_h + 1;
    args.num_cols = args.out_w / args.in_h + 1;
    args.h_offset = (args.in_h * args.num_rows - args.out_h) / (args.num_rows - 1);
    args.w_offset = (args.in_w * args.num_cols - args.out_w) / (args.num_cols - 1);
    args.h_extra_offset = (args.in_h * args.num_rows - args.out_h) % (args.num_rows - 1);
    args.w_extra_offset = (args.in_w * args.num_cols - args.out_w) % (args.num_cols - 1);

    args.h_len = 104;
    args.w_len = 104;
    args.im_data = orig.data;
    args.threads = 48;
    args.n = 48;
    args.d = &buffer;
    args.type = ESPCN_DEMO_DATA;

    // float *im_data = calloc(args.out_c*args.out_h*args.out_w, sizeof(float));
    // args.im_data = im_data;


    printf("%d, %d, %d, %d, %d, %d\n\n", args.num_rows, args.num_cols, args.h_offset, args.w_offset, args.h_extra_offset, args.w_extra_offset);
    
    // pthread_t load_thread = load_data_espcn(args);
    // load_data_espcn(args);
    pthread_t load_thread = load_data_espcn(args);

    pthread_join(load_thread, 0);
    // sleep(1);
    double time=what_time_is_it_now();
    for(int t=0; t<1000; t++){

        network_predict_data(net, buffer);
    }
    printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);



    image im = data2im(args);
    // save_image(im, "test_data/test1233.jpg");
    free_image(im);

    return;
}
 

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
    if(0==strcmp(argv[2], "data_test")) data_test(datacfg, cfg, weights, filename, gpus, ngpus, clear);
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
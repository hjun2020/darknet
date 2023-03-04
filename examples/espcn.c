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
image enhence_image2(char *filename, network *net,int channels, int out_w, int out_h, int in_w, int in_h)
{
    int w,h,c;
    unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
    if (!data) {
        fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n", filename, stbi_failure_reason());
        exit(0);
    }
    images images;
    in_w = w;
    in_h = h;
    int w_remainder = in_w % out_w;
    int h_remainder = in_h % out_h; 
    int row;
    int col;
    // if (w_remainder != 0){
    //     col = in_w / out_w + 1;
    // } else {
    //     col = in_w / out_w;
    // }
    // if (h_remainder != 0){
    //     row = in_h / out_h + 1;
    // } else {
    //     row = in_h / out_h;
    // }
    col = in_w / out_w + 2;
    row = in_h / out_h + 2;

    int row_offset = (row * out_h - in_h) / (row-1);
    int col_offset = (col * out_w - in_w) / (col-1);

    int row_remainder = (row * out_h - in_h) % (row-1);
    int col_remainder = (col * out_w - in_w) % (col-1);
    images.data = calloc(row * col, sizeof(image));
    images.row = row;
    images.col = col;
    
    int w_len = out_w;
    int h_len = out_h;
    images.w = 3*w_len;
    images.h = 3*h_len;
    printf("row_offset: %d, col_offset: %d, row_remainder: %d, col_remainder: %d\n", row_offset, col_offset, row_remainder, col_remainder);

    for (int i=0; i<row; i++){
        for (int j=0; j<col; j++) {
            int r_offset = i*row_offset;
            int c_offset = j*col_offset;
            if(i == row-1){
                r_offset += row_remainder;
            }
            if(j == col-1){
                c_offset += col_remainder;
            }

            // printf("out_w: %d, out_h: %d\n", 3*out_w, 3*out_h);
            image partial_img = load_partial_image_stb(data, 3, j*w_len-c_offset, w_len, i*h_len-r_offset, h_len, in_w, in_h, c);
            image temp_image = make_image(3*out_w, 3*out_h, 3);
            temp_image.data = network_predict(net, partial_img.data);
            images.data[i*col+j] = copy_image(temp_image);
            
            printf("row: %d, col: %d, w_range: %d %d, h_range: %d %d\n", i, j, j*w_len-c_offset, w_len, i*h_len-r_offset, h_len);
        }
    }
    free(data);
    // save_image(images.data[2], "temp");
    image out_im = make_image(3*in_w, 3*in_h, 3);

    printf("row: %d, col: %d, in_w: %d, in_h: %d, row_remainder: %d, col_remainder: %d\n", row, col, in_w, in_h, row_remainder, col_remainder);
    printf("row_offset: %d, col_offset: %d\n", row_offset, col_offset);
    row_offset *= 3;
    col_offset *= 3;
    printf("row_offset: %d, col_offset: %d\n", row_offset, col_offset);

    row_remainder *= 3;
    col_remainder *= 3;

    // printf("row: %d, col: %d, in_w: %d, in_h: %d, row_offset: %d, col_offset: %d, row_remainder: %d, col_remainder: %d\n", row, col, in_w, in_h, row_offset, col_offset, row_remainder, col_remainder);

    int edge_offset=10;
    for(int c=0; c<channels; c++){
        for(int i=0; i<row * (out_h*3) ; i++){
            for(int j=0; j<col * (out_w*3); j++){
                int row_idx = i / (out_h*3);
                int col_idx = j / (out_w*3);
                int w_idx = j % (out_w*3);
                int h_idx = i % (out_h*3);

                int r_offset = row_idx*(row_offset);
                int c_offset = col_idx*(col_offset);

                if(row_idx == row-1){
                    r_offset += row_remainder;
                }
                if(col_idx == col-1){
                    c_offset += col_remainder;
                }
                if (row_idx != 0 && h_idx < 24){
                    continue;
                }
                if (col_idx != 0 && w_idx < 24){
                    continue;
                }

                out_im.data[(j-c_offset) + (i-r_offset)*(in_w*3) + (in_w*3)*(in_h*3)*c] 
                = images.data[row_idx*col+col_idx].data[(h_idx)*(out_w*3) 
                + w_idx + c*(out_h*3)*(out_w*3)];
            }
        }

    }
    return out_im;

}
// image enhence_image(char *filename, network *net,int channels, int out_w, int out_h, int in_w, int in_h)
// {
//     printf("image enhencer started");
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

//             printf("out_w: %d, out_h: %d", 3*out_w, 3*out_h);
//             image partial_img = load_partial_image_stb(filename, 3, j*w_len-r_offset, w_len, i*h_len-c_offset, h_len);
//             image temp_image = make_image(3*out_w, 3*out_h, 3);
//             temp_image.data = network_predict(net, partial_img.data);
//             images.data[i*col+j] = copy_image(temp_image);
//             // printf("row: %d, col: %d, w_range: %d %d, h_range: %d %d\n", i, j, j*w_len-r_offset, w_len, i*h_len-c_offset, h_len);
//         }
//     }
//     image out_im = make_image(3*in_w, 3*in_h, 3);
//     // int row = images.row;
//     // int col = images.col;
//     // int in_w = images.w;
//     // int in_h = images.h;
//     printf("row: %d, col: %d, in_w: %d, in_h: %d", row, col, in_w, in_h);


//     row_offset *= 3;
//     col_offset *= 3;

//     row_remainder *= 3;
//     col_remainder *= 3;

//     // printf("row: %d, col: %d, in_w: %d, in_h: %d, row_offset: %d, col_offset: %d, row_remainder: %d, col_remainder: %d\n", row, col, in_w, in_h, row_offset, col_offset, row_remainder, col_remainder);

//     int edge_offset = 5;
//     for(int c=0; c<channels; c++){
//         for(int i=0; i<row * (in_w*3) ; i++){
//             for(int j=0; j<col * (in_h*3); j++){
//                 int row_idx = i / (out_h*3);
//                 int col_idx = j / (out_w*3);
//                 int w_idx = j % (out_w*3);
//                 int h_idx = i % (out_h*3);

//                 int r_offset = row_idx*(row_offset);
//                 int c_offset = col_idx*(col_offset);

//                 if(row_idx == row-1){
//                     r_offset += row_remainder;
//                 }
//                 if(col_idx == col-1){
//                     c_offset += col_remainder;
//                 }



//                 if (row_idx != 0 && h_idx < 3){
//                     continue;
//                 }
//                 if (col_idx != 0 && w_idx < 3){
//                     continue;
//                 }
//                 // if (row_idx != row-1 && h_idx > in_h-3){
//                 //     continue;
//                 // }
//                 // if (col_idx != col-1 && w_idx > in_w-3){
//                 //     continue;
//                 // }
//                 out_im.data[(j-c_offset) + (i-r_offset)*(in_w*3) + (in_w*3)*(in_h*3)*c] 
//                 = images.data[row_idx*col+col_idx].data[(h_idx)*out_w 
//                 + w_idx + c*out_h*out_w];
//             }
//         }

//     }
//     return out_im;
// }

// void print_enhencer_enhencement(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h)
// {
//     int i, j;
//     for(i = 0; i < total; ++i){
//         float xmin = dets[i].bbox.x - dets[i].bbox.w/2. + 1;
//         float xmax = dets[i].bbox.x + dets[i].bbox.w/2. + 1;
//         float ymin = dets[i].bbox.y - dets[i].bbox.h/2. + 1;
//         float ymax = dets[i].bbox.y + dets[i].bbox.h/2. + 1;

//         if (xmin < 1) xmin = 1;
//         if (ymin < 1) ymin = 1;
//         if (xmax > w) xmax = w;
//         if (ymax > h) ymax = h;

//         for(j = 0; j < classes; ++j){
//             if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
//                     xmin, ymin, xmax, ymax);
//         }
//     }
// }

// void print_imagenet_enhencement(FILE *fp, int id, detection *dets, int total, int classes, int w, int h)
// {
//     int i, j;
//     for(i = 0; i < total; ++i){
//         float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
//         float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
//         float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
//         float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

//         if (xmin < 0) xmin = 0;
//         if (ymin < 0) ymin = 0;
//         if (xmax > w) xmax = w;
//         if (ymax > h) ymax = h;

//         for(j = 0; j < classes; ++j){
//             int class = j;
//             if (dets[i].prob[class]) fprintf(fp, "%d %d %f %f %f %f %f\n", id, j+1, dets[i].prob[class],
//                     xmin, ymin, xmax, ymax);
//         }
//     }
// }

// void validate_enhencer_flip(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
// {
//     int j;
//     list *options = read_data_cfg(datacfg);
//     char *valid_images = option_find_str(options, "valid", "data/train.list");
//     char *name_list = option_find_str(options, "names", "data/names.list");
//     char *prefix = option_find_str(options, "results", "results");
//     char **names = get_labels(name_list);
//     char *mapf = option_find_str(options, "map", 0);
//     int *map = 0;
//     if (mapf) map = read_map(mapf);

//     network *net = load_network(cfgfile, weightfile, 0);
//     set_batch_network(net, 2);
//     fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
//     srand(time(0));

//     list *plist = get_paths(valid_images);
//     char **paths = (char **)list_to_array(plist);

//     layer l = net->layers[net->n-1];
//     int classes = l.classes;

//     char buff[1024];
//     char *type = option_find_str(options, "eval", "voc");
//     FILE *fp = 0;
//     FILE **fps = 0;
//     int coco = 0;
//     int imagenet = 0;
//     if(0==strcmp(type, "coco")){
//         if(!outfile) outfile = "coco_results";
//         snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
//         fp = fopen(buff, "w");
//         fprintf(fp, "[\n");
//         coco = 1;
//     } else if(0==strcmp(type, "imagenet")){
//         if(!outfile) outfile = "imagenet-detection";
//         snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
//         fp = fopen(buff, "w");
//         imagenet = 1;
//         classes = 200;
//     } else {
//         if(!outfile) outfile = "comp4_det_test_";
//         fps = calloc(classes, sizeof(FILE *));
//         for(j = 0; j < classes; ++j){
//             snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
//             fps[j] = fopen(buff, "w");
//         }
//     }

//     int m = plist->size;
//     int i=0;
//     int t;

//     float thresh = .005;
//     float nms = .45;

//     int nthreads = 4;
//     image *val = calloc(nthreads, sizeof(image));
//     image *val_resized = calloc(nthreads, sizeof(image));
//     image *buf = calloc(nthreads, sizeof(image));
//     image *buf_resized = calloc(nthreads, sizeof(image));
//     pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

//     image input = make_image(net->w, net->h, net->c*2);

//     load_args args = {0};
//     args.w = net->w;
//     args.h = net->h;
//     //args.type = IMAGE_DATA;
//     args.type = LETTERBOX_DATA;

//     for(t = 0; t < nthreads; ++t){
//         args.path = paths[i+t];
//         args.im = &buf[t];
//         args.resized = &buf_resized[t];
//         thr[t] = load_data_in_thread(args);
//     }
//     double start = what_time_is_it_now();
//     for(i = nthreads; i < m+nthreads; i += nthreads){
//         fprintf(stderr, "%d\n", i);
//         for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
//             pthread_join(thr[t], 0);
//             val[t] = buf[t];
//             val_resized[t] = buf_resized[t];
//         }
//         for(t = 0; t < nthreads && i+t < m; ++t){
//             args.path = paths[i+t];
//             args.im = &buf[t];
//             args.resized = &buf_resized[t];
//             thr[t] = load_data_in_thread(args);
//         }
//         for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
//             char *path = paths[i+t-nthreads];
//             char *id = basecfg(path);
//             copy_cpu(net->w*net->h*net->c, val_resized[t].data, 1, input.data, 1);
//             flip_image(val_resized[t]);
//             copy_cpu(net->w*net->h*net->c, val_resized[t].data, 1, input.data + net->w*net->h*net->c, 1);

//             network_predict(net, input.data);
//             int w = val[t].w;
//             int h = val[t].h;
//             int num = 0;
//             detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0, &num);
//             if (nms) do_nms_sort(dets, num, classes, nms);
//             if (coco){
//                 print_cocos(fp, path, dets, num, classes, w, h);
//             } else if (imagenet){
//                 print_imagenet_enhencement(fp, i+t-nthreads+1, dets, num, classes, w, h);
//             } else {
//                 print_enhencer_enhencement(fps, id, dets, num, classes, w, h);
//             }
//             free_detections(dets, num);
//             free(id);
//             free_image(val[t]);
//             free_image(val_resized[t]);
//         }
//     }
//     for(j = 0; j < classes; ++j){
//         if(fps) fclose(fps[j]);
//     }
//     if(coco){
//         fseek(fp, -2, SEEK_CUR); 
//         fprintf(fp, "\n]\n");
//         fclose(fp);
//     }
//     fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
// }


// void validate_enhencer(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
// {
//     int j;
//     list *options = read_data_cfg(datacfg);
//     char *valid_images = option_find_str(options, "valid", "data/train.list");
//     char *name_list = option_find_str(options, "names", "data/names.list");
//     char *prefix = option_find_str(options, "results", "results");
//     char **names = get_labels(name_list);
//     char *mapf = option_find_str(options, "map", 0);
//     int *map = 0;
//     if (mapf) map = read_map(mapf);

//     network *net = load_network(cfgfile, weightfile, 0);
//     set_batch_network(net, 1);
//     fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
//     srand(time(0));

//     list *plist = get_paths(valid_images);
//     char **paths = (char **)list_to_array(plist);

//     layer l = net->layers[net->n-1];
//     int classes = l.classes;

//     char buff[1024];
//     char *type = option_find_str(options, "eval", "voc");
//     FILE *fp = 0;
//     FILE **fps = 0;
//     int coco = 0;
//     int imagenet = 0;
//     if(0==strcmp(type, "coco")){
//         if(!outfile) outfile = "coco_results";
//         snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
//         fp = fopen(buff, "w");
//         fprintf(fp, "[\n");
//         coco = 1;
//     } else if(0==strcmp(type, "imagenet")){
//         if(!outfile) outfile = "imagenet-detection";
//         snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
//         fp = fopen(buff, "w");
//         imagenet = 1;
//         classes = 200;
//     } else {
//         if(!outfile) outfile = "comp4_det_test_";
//         fps = calloc(classes, sizeof(FILE *));
//         for(j = 0; j < classes; ++j){
//             snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
//             fps[j] = fopen(buff, "w");
//         }
//     }


//     int m = plist->size;
//     int i=0;
//     int t;

//     float thresh = .005;
//     float nms = .45;

//     int nthreads = 4;
//     image *val = calloc(nthreads, sizeof(image));
//     image *val_resized = calloc(nthreads, sizeof(image));
//     image *buf = calloc(nthreads, sizeof(image));
//     image *buf_resized = calloc(nthreads, sizeof(image));
//     pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

//     load_args args = {0};
//     args.w = net->w;
//     args.h = net->h;
//     //args.type = IMAGE_DATA;
//     args.type = LETTERBOX_DATA;

//     for(t = 0; t < nthreads; ++t){
//         args.path = paths[i+t];
//         args.im = &buf[t];
//         args.resized = &buf_resized[t];
//         thr[t] = load_data_in_thread(args);
//     }
//     double start = what_time_is_it_now();
//     for(i = nthreads; i < m+nthreads; i += nthreads){
//         fprintf(stderr, "%d\n", i);
//         for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
//             pthread_join(thr[t], 0);
//             val[t] = buf[t];
//             val_resized[t] = buf_resized[t];
//         }
//         for(t = 0; t < nthreads && i+t < m; ++t){
//             args.path = paths[i+t];
//             args.im = &buf[t];
//             args.resized = &buf_resized[t];
//             thr[t] = load_data_in_thread(args);
//         }
//         for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
//             char *path = paths[i+t-nthreads];
//             char *id = basecfg(path);
//             float *X = val_resized[t].data;
//             network_predict(net, X);
//             int w = val[t].w;
//             int h = val[t].h;
//             int nboxes = 0;
//             detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0, &nboxes);
//             if (nms) do_nms_sort(dets, nboxes, classes, nms);
//             if (coco){
//                 print_cocos(fp, path, dets, nboxes, classes, w, h);
//             } else if (imagenet){
//                 print_imagenet_enhencement(fp, i+t-nthreads+1, dets, nboxes, classes, w, h);
//             } else {
//                 print_enhencer_enhencement(fps, id, dets, nboxes, classes, w, h);
//             }
//             free_detections(dets, nboxes);
//             free(id);
//             free_image(val[t]);
//             free_image(val_resized[t]);
//         }
//     }
//     for(j = 0; j < classes; ++j){
//         if(fps) fclose(fps[j]);
//     }
//     if(coco){
//         fseek(fp, -2, SEEK_CUR); 
//         fprintf(fp, "\n]\n");
//         fclose(fp);
//     }
//     fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
// }

// void validate_enhencer_recall(char *cfgfile, char *weightfile)
// {
//     network *net = load_network(cfgfile, weightfile, 0);
//     set_batch_network(net, 1);
//     fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
//     srand(time(0));

//     list *plist = get_paths("data/coco_val_5k.list");
//     char **paths = (char **)list_to_array(plist);

//     layer l = net->layers[net->n-1];

//     int j, k;

//     int m = plist->size;
//     int i=0;

//     float thresh = .001;
//     float iou_thresh = .5;
//     float nms = .4;

//     int total = 0;
//     int correct = 0;
//     int proposals = 0;
//     float avg_iou = 0;

//     for(i = 0; i < m; ++i){
//         char *path = paths[i];
//         image orig = load_image_color(path, 0, 0);
//         image sized = resize_image(orig, net->w, net->h);
//         char *id = basecfg(path);
//         network_predict(net, sized.data);
//         int nboxes = 0;
//         detection *dets = get_network_boxes(net, sized.w, sized.h, thresh, .5, 0, 1, &nboxes);
//         if (nms) do_nms_obj(dets, nboxes, 1, nms);

//         char labelpath[4096];
//         find_replace(path, "images", "labels", labelpath);
//         find_replace(labelpath, "JPEGImages", "labels", labelpath);
//         find_replace(labelpath, ".jpg", ".txt", labelpath);
//         find_replace(labelpath, ".JPEG", ".txt", labelpath);

//         int num_labels = 0;
//         box_label *truth = read_boxes(labelpath, &num_labels);
//         for(k = 0; k < nboxes; ++k){
//             if(dets[k].objectness > thresh){
//                 ++proposals;
//             }
//         }
//         for (j = 0; j < num_labels; ++j) {
//             ++total;
//             box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
//             float best_iou = 0;
//             for(k = 0; k < l.w*l.h*l.n; ++k){
//                 float iou = box_iou(dets[k].bbox, t);
//                 if(dets[k].objectness > thresh && iou > best_iou){
//                     best_iou = iou;
//                 }
//             }
//             avg_iou += best_iou;
//             if(best_iou > iou_thresh){
//                 ++correct;
//             }
//         }

//         fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
//         free(id);
//         free_image(orig);
//         free_image(sized);
//     }
// }

// void test2(char *cfgfile, char *weightfile){
//     network *net = load_network(cfgfile, weightfile, 0);
// }

 
void test_enhencer(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{   

    // list *options = read_data_cfg(datacfg);
    // char *name_list = option_find_str(options, "names", "data/names.list");
    // char **names = get_labels(name_list);

    // image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;


    float nms=.45;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        save_image(im, "loaded_image");
        // image sized = resize_image(im, net->w, net->h);
        // save_image(sized, "input_image");
        // image sized = letterbox_image(im, net->w, net->h);
        //image sized = resize_image(im, net->w, net->h);
        //image sized2 = resize_max(im, net->w);
        //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
        //resize_network(net, sized.w, sized.h);
        layer l = net->layers[net->n-1];


        // float *X = sized.data;
        time=what_time_is_it_now();
        
        // float *predicted_output = network_predict(net, X);

        printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
        image out2 = enhence_image2(input, net, 3, 104, 104, 700, 700);
        // save_image(out2, "/home/ubuntu/developer/django-project/src/media/output_image2");
        save_image(out2, outfile);
        // image output_im;
        // output_im.w = 3*net->w;
        // output_im.h = 3*net->h;
        // output_im.c = 3;
        // output_im.data = predicted_output;
        // images out_image = load_partial_enhenced_images_stb(input, net, 3, 104, 104, 700, 500);
        // image out = merge_partial_images(out_image, 3, 2100, 1500);
        // save_image(out, "output_image");

        return;


        // int nboxes = 0;
        // detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
        // //printf("%d\n", nboxes);
        // //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        // if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        // draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
        // free_detections(dets, nboxes);
        // if(outfile){
        //     save_image(im, outfile);
        // }
        // else{
        //     save_image(im, "predictions");
#ifdef OPENCV
            make_window("predictions", 512, 512, 0);
            show_image(im, "predictions", 0);
#endif
    //     }

    //     free_image(im);
    //     free_image(sized);
    //     if (filename) break;
    }
}


void run_enhencer(int argc, char **argv)
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
    if(0==strcmp(argv[2], "test")) test_enhencer(datacfg, cfg, weights, filename, thresh, hier_thresh, outfile, fullscreen);
    else if(0==strcmp(argv[2], "train")) train_enhencer(datacfg, cfg, weights, gpus, ngpus, clear);
}
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
 



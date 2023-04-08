#ifdef OPENCV

#include "stdio.h"
#include "stdlib.h"
#include "opencv2/opencv.hpp"
#include "image.h"


#include "opencv2/core/core_c.h"
#include "opencv2/videoio/legacy/constants_c.h"
#include "opencv2/highgui/highgui_c.h"

using namespace cv;

extern "C" {

IplImage *image_to_ipl(image im)
{
    int x,y,c;
    IplImage *disp = cvCreateImage(cvSize(im.w,im.h), IPL_DEPTH_8U, im.c);
    int step = disp->widthStep;
    for(y = 0; y < im.h; ++y){
        for(x = 0; x < im.w; ++x){
            for(c= 0; c < im.c; ++c){
                float val = im.data[c*im.h*im.w + y*im.w + x];
                disp->imageData[y*step + x*im.c + c] = (unsigned char)(val*255);
            }
        }
    }
    return disp;
}

image ipl_to_image(IplImage* src)
{
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)src->imageData;
    int step = src->widthStep;
    int i, j, k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
    return im;
}

Mat image_to_mat(image im)
{
    image copy = copy_image(im);
    constrain_image(copy);
    if(im.c == 3) rgbgr_image(copy);

    IplImage *ipl = image_to_ipl(copy);
    Mat m = cvarrToMat(ipl, true);
    cvReleaseImage(&ipl);
    free_image(copy);
    return m;
}


//added for espcn TEMP!!!!!!!!!!!!
image mat_to_image_test(void *ptr)
{
    Mat *m = (Mat *) ptr;
    IplImage ipl = cvIplImage(*m);
    image im = ipl_to_image(&ipl);
    rgbgr_image(im);
    save_image(im, "data_test/TTTSET");
    return im;

}

image mat_to_image(Mat m)
{
    // IplImage ipl = m;
    IplImage ipl = cvIplImage(m);
    image im = ipl_to_image(&ipl);
    rgbgr_image(im);
    return im;
}

image mat_to_image_single_channel(Mat m)
{
    // IplImage ipl = m;
    IplImage ipl = cvIplImage(m);
    image im = ipl_to_image(&ipl);
    // rgbgr_image(im);
    return im;
}



void *open_video_stream(const char *f, int c, int w, int h, int fps)
{
    VideoCapture *cap;
    if(f) cap = new VideoCapture(f);
    else cap = new VideoCapture(c);
    if(!cap->isOpened()) return 0;
    if(w) cap->set(CV_CAP_PROP_FRAME_WIDTH, w);
    if(h) cap->set(CV_CAP_PROP_FRAME_HEIGHT, w);
    if(fps) cap->set(CV_CAP_PROP_FPS, w);
    return (void *) cap;
}

void *get_mat_from_stream(void *p)
{
    VideoCapture *cap = (VideoCapture *)p;
    Mat m;
    *cap >> m;

    Mat ycrcb;
    cvtColor(m, ycrcb, COLOR_BGR2YCrCb);
    
    Mat *ycrcb_channels = (Mat *)calloc(3, sizeof(Mat));
    split(ycrcb, ycrcb_channels);

    // Mat *ycrcb_channels = (Mat *)calloc(1, sizeof(Mat));
    // ycrcb_channels[0] = m;
    return ycrcb_channels;
}

image get_image_from_stream(void *p)
{
    VideoCapture *cap = (VideoCapture *)p;
    Mat m;
    *cap >> m;
    if(m.empty()) return make_empty_image(0,0,0);
    return mat_to_image(m);
}

image load_image_cv(char *filename, int channels)
{
    int flag = -1;
    if (channels == 0) flag = -1;
    else if (channels == 1) flag = 0;
    else if (channels == 3) flag = 1;
    else {
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    }
    Mat m;
    m = imread(filename, flag);
    if(!m.data){
        fprintf(stderr, "Cannot load image \"%s\"\n", filename);
        char buff[256];
        sprintf(buff, "echo %s >> bad.list", filename);
        system(buff);
        return make_image(10,10,3);
        //exit(0);
    }
    image im = mat_to_image(m);
    return im;
}

int show_image_cv(image im, const char* name, int ms)
{
    Mat m = image_to_mat(im);
    imshow(name, m);
    int c = waitKey(ms);
    if (c != -1) c = c%256;
    return c;
}

void make_window(char *name, int w, int h, int fullscreen)
{
    namedWindow(name, WINDOW_NORMAL); 
    if (fullscreen) {
        setWindowProperty(name, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    } else {
        resizeWindow(name, w, h);
        if(strcmp(name, "Demo") == 0) moveWindow(name, 0, 0);
    }
}


image extract_luminance2(void *data)
{
    Mat *ptr = (Mat *) data;
    // int flag = -1;
    // if (channels == 0) flag = -1;
    // else if (channels == 1) flag = 0;
    // else if (channels == 3) flag = 1;
    // else {
    //     fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    // }
    // Mat m = imread(filename);

    // Mat ycrcb;
    // cvtColor(m, ycrcb, COLOR_BGR2YCrCb);

    // Mat ycrcb_channels[3];
    // split(ycrcb, ycrcb_channels);
    image im = mat_to_image_single_channel(ptr[0]);

    return im;
}

image extract_luminance(char *filename, int channels)
{
    int flag = -1;
    if (channels == 0) flag = -1;
    else if (channels == 1) flag = 0;
    else if (channels == 3) flag = 1;
    else {
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    }
    Mat m = imread(filename);

    Mat ycrcb;
    cvtColor(m, ycrcb, COLOR_BGR2YCrCb);

    Mat ycrcb_channels[3];
    split(ycrcb, ycrcb_channels);
    image im = mat_to_image_single_channel(ycrcb_channels[0]);

    return im;
}



void *rgb2ycbcr(char *filename)
{
    Mat img1 = imread(filename);
    Mat img;
    Size size1(200, 200); // set the output size
    resize(img1, img, size1, 0, 0, cv::INTER_LINEAR); 

    Size size2(600,600);

    Mat ycrcb;
    cvtColor(img, ycrcb, COLOR_BGR2YCrCb);


    Mat ycrcb_channels[3];
    split(ycrcb, ycrcb_channels);


    // Mat ycrcb_channels_resized[3];
    // image net_input = mat_to_image_single_channel(ycrcb_channels[0]);
    // image net_output = resize_image(net_input, 700, 700);
    

    Mat *outputs = (Mat *)calloc(3, sizeof(Mat));

    resize(ycrcb_channels[0], outputs[0], size2, 0, 0, cv::INTER_LINEAR); 
    resize(ycrcb_channels[1], outputs[1], size2, 0, 0, cv::INTER_LINEAR); 
    resize(ycrcb_channels[2], outputs[2], size2, 0, 0, cv::INTER_LINEAR); 

    Mat output;
    Mat output2;

    merge(outputs,3, output);
    cvtColor(output, output2, COLOR_YCrCb2RGB);

    image outimg33 = mat_to_image(output2);
    rgbgr_image(outimg33);
    save_image(outimg33, "data_test/opencv_simple_resize");

    return outputs;

}

void merge_ycbcr2rgb(void *data, image im)
{
    // merge(outputs,3, output);
    Mat *ptr = (Mat *) data;
    Mat m = image_to_mat(im);

    Mat output[3];
    output[0] = image_to_mat(im);
    output[1] = ptr[1];
    output[2] = ptr[2];

    Mat output2;
    merge(output,3, output2);
    Mat output3;
    cvtColor(output2, output3, COLOR_YCrCb2RGB);

    image res = mat_to_image(output3);
    rgbgr_image(res);

    

    // save_image(res, "data_test/napoli");



}

void ycbcr2rgb(void *data)
{
    Mat *ptr = (Mat *) data;
    image temp = mat_to_image_single_channel(ptr[0]);
    // printf("DOEN!!!\n");
    save_image(temp, "data_test/napoli");
    // Mat output;
    // Mat output2;
    // merge(ptr,3, output);
    // cvtColor(ptr, output2, COLOR_YCrCb2RGB);

    return;

}



}

#endif

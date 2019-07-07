#include <stdio.h>
#include <stdlib.h>
#include <iostream>

extern "C" {
    #include "sod.h"
}

__global__ void mean_filter(float *image, int window_size, int frame_size, int new_frame_size, float *new_image) {
    long index = (blockDim.x * blockIdx.x) + threadIdx.x;
    int i = index / new_frame_size;
    int j = index % new_frame_size;
        
    if (i < new_frame_size & j < new_frame_size) {
        float total = 0.0f;
        for(int a = 0; a < window_size; a++){
            for(int b = 0; b < window_size; b++){
                total += image[frame_size * (i + a) + j + b];
            }    
        }
        new_image[new_frame_size * i + j] = total / (window_size * window_size);
    }
}

void mean_filter_h(float *image, int window_size, int frame_size, int new_frame_size, float *new_image) {
    for(int i = 0; i < new_frame_size; i++) {
        for(int j = 0; j < new_frame_size; j++) {
            float total = 0.0f;
            for(int a = 0; a < window_size; a++){
                for(int b = 0; b < window_size; b++){
                    total += image[frame_size * (i + a) + j + b];
                }    
            }
            new_image[new_frame_size * i + j] = total / (window_size * window_size);
        }
    }
}

int main(int argc,char **argv) {
    int window_size = atoll(argv[2] ? argv[2]: "3");
    int frame_size = atoll(argv[1] ? argv[1]: "1200");

    printf("Image size - %d | Window size - %d\n", frame_size, window_size);

    // calculate the new frame size and count of its element
    int new_frame_size = frame_size - window_size + 1;
    long max_length = new_frame_size * new_frame_size;

    // image input and output paths
    const char *file_path = "./images/image.jpeg";
    const char *file_out_path = "./images/image_cpu.png";
    const char *file_out_dev_path = "./images/image_gpu.png";
    const char *file_bw_out_path = "./images/image_bw.png";

    // load the image and resize into given frame size
    sod_img image = sod_img_load_from_file(file_path, SOD_IMG_GRAYSCALE);
    image = sod_resize_image(image, frame_size, frame_size);

    // create empyt images to store cpu and gpu processed images
    sod_img image_out = sod_make_image(new_frame_size, new_frame_size, 1);
    sod_img image_out_dev = sod_make_image(new_frame_size, new_frame_size, 1);

    // define block size and number of blocks
    int block_size = 1024;
    int block_no = (max_length / 1024) + ((max_length % 1024 > 0) ? 1 : 0);

    dim3 dimBlock(block_size,1,1);
    dim3 dimGrid(block_no,1,1);

    // declare points to access device memory
    float *image_data_dev;
    float *image_data_out_dev;

    // allocating device memory 
    printf("Allocating device memory..\n");
    cudaMalloc(&image_data_dev, sizeof(float) * frame_size * frame_size);
    cudaMalloc(&image_data_out_dev, sizeof(float) * new_frame_size * new_frame_size);

    // copy image data from host to device
    printf("Copying to device..\n");
    cudaMemcpy(image_data_dev, image.data, sizeof(float) * frame_size * frame_size, cudaMemcpyHostToDevice);

    // do the gpu calculations
    printf("Doing GPU calculation..\n");
    clock_t start_gpu=clock();
    mean_filter<<<block_no,block_size>>>(image_data_dev, window_size, frame_size, new_frame_size, image_data_out_dev);
    cudaThreadSynchronize();
    clock_t end_gpu = clock();
    
    // do the cpu calculations
    printf("Doing CPU Calculation..\n");
    clock_t start_cpu = clock();
    mean_filter_h(image.data, window_size, frame_size, new_frame_size, image_out.data);
    clock_t end_cpu = clock();

    // calculate the time for calculations
    double time_gpu = (double)(end_gpu - start_gpu)/CLOCKS_PER_SEC;
    double time_cpu = (double)(end_cpu - start_cpu)/CLOCKS_PER_SEC;
    
    // copy the output image from device memory to host
    printf("Copying to host..\n");
    cudaMemcpy(image_out_dev.data, image_data_out_dev, sizeof(float) * new_frame_size * new_frame_size, cudaMemcpyDeviceToHost);

    // save processed images
    printf("Saving images..\n");
    sod_img_save_as_png(image, file_bw_out_path);
    sod_img_save_as_png(image_out, file_out_path);
    sod_img_save_as_png(image_out_dev, file_out_dev_path);

    printf("GPU Time: %f, CPU Time: %f\n", time_gpu, time_cpu);

    // free the device and host memory
    sod_free_image(image_out_dev);
    sod_free_image(image_out);
    sod_free_image(image);
    cudaFree(image_data_out_dev);
    cudaFree(image_data_dev);

    return 0;
}
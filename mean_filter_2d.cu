#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cuda.h>
#include "sod.h"
#define MAX_THREADS 1024

__global__ void mean_filter(int **image, int window_size, int frame_size, int **new_image) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    int total = 0;
    for(int a = 0; a < window_size; a++){
        for(int b = 0; b < window_size; b++){
            total += image[i + a][j + b];
        }    
    }
    new_image[i][j] =  total / (window_size * window_size);
}

void mean_filter_h(int **image, int window_size, int frame_size, int **new_image) {
    int new_length = frame_size - window_size + 1;
    for(int i = 0; i < new_length; i++) {
        for(int j = 0; j < new_length; j++) {
            int total = 0;
            for(int a = 0; a < window_size; a++){
                for(int b = 0; b< window_size; b++){
                    total += image[i + a][j + b];
                }    
            }
            new_image[i][j] =  total / (window_size * window_size);
        }
    }
}

int** init_2d_array(int nrow, int ncol) {
    int **arr = (int **)malloc(nrow * sizeof(int *));

    for(int i = 0; i < nrow; i++) {
        arr[i] = (int *)malloc(ncol * sizeof(int)); 
    }

    return arr;
}

void print_2d_array(int **arr, int nrow, int ncol) {
    for (int i = 0; i <  nrow; i++) {
        for (int j = 0; j < ncol; j++) {
            printf("%d ", arr[i][j]);
        }
        printf("\n");
    }
}

void randomize_2d_array(int **arr, int nrow, int ncol) {
    for (int i = 0; i <  nrow; i++) {
        for (int j = 0; j < ncol; j++) {
            arr[i][j] = rand() % 256;
        }
    }
}

int main(int argc,char **argv) {
    printf("MAX Threads: %d\n", MAX_THREADS);
    int window_size = 3;
    int frame_size = 10;
    int new_frame_size = frame_size - window_size + 1;

    int **image = init_2d_array(frame_size, frame_size);
    int **image_new = init_2d_array(new_frame_size, new_frame_size);
    int **image_new_2 = init_2d_array(new_frame_size, new_frame_size);

    randomize_2d_array(image, frame_size, frame_size);

    int block_size = new_frame_size;

    if (new_frame_size > MAX_THREADS ) {
        block_size = new_frame_size / 2;
    }
    
    int block_no = new_frame_size;
    dim3 dimBlock(block_size,1,1);
    dim3 dimGrid(block_no,1,1);

    int **image_host = (int **) malloc(frame_size * sizeof(int *));
    int **image_new_host = (int **) malloc(new_frame_size * sizeof(int *));
    int **image_dev;
    int **image_new_dev;

    printf("Allocating device memory on host and copy..\n");

    for(int i = 0; i < frame_size; i++) {
        cudaMalloc(&image_host[i], frame_size * sizeof(int));
        cudaMemcpy(image_host[i], image[i], frame_size * sizeof(int), cudaMemcpyHostToDevice);
    }

    for(int i = 0; i < new_frame_size; i++) {
        cudaMalloc(&image_new_host[i], new_frame_size * sizeof(int));
    }

    cudaMalloc(&image_dev, frame_size * sizeof(int *));
    cudaMalloc(&image_new_dev, new_frame_size * sizeof(int *));
    
    cudaMemcpy(image_dev, image_host, frame_size * sizeof(int *), cudaMemcpyHostToDevice);
    cudaMemcpy(image_new_dev, image_new_host, new_frame_size * sizeof(int *), cudaMemcpyHostToDevice);
    
    printf("Completed copying to device and memory allocation..\n");

    clock_t start_gpu=clock();

    printf("Doing GPU calculation..\n");

    mean_filter<<<block_no,block_size>>>(image_dev, window_size, frame_size, image_new_dev);
   
    cudaThreadSynchronize();

    clock_t end_gpu = clock();
    clock_t start_cpu = clock();
    
    printf("Doing CPU Vector add\n");

    mean_filter_h(image, window_size, frame_size, image_new);

    clock_t end_cpu = clock();
   
    double time_gpu = (double)(end_gpu - start_gpu)/CLOCKS_PER_SEC;
    double time_cpu = (double)(end_cpu - start_cpu)/CLOCKS_PER_SEC;

    cudaMemcpy(image_new_host, image_new_dev, new_frame_size * sizeof(int *), cudaMemcpyDeviceToHost);

    for(int i = 0; i < new_frame_size; i++) {
        cudaMemcpy(image_new_2[i], image_new_host[i], new_frame_size * sizeof(int), cudaMemcpyDeviceToHost);
    }

    printf("Frame Size: %d, Window Size: %d, GPU Time: %f, CPU Time: %f\n", frame_size, window_size, time_gpu, time_cpu);

    cudaFree(image_dev);
    cudaFree(image_new_dev);
    free(image_host);
    free(image_new_host);
    free(image);
    free(image_new);
    free(image_new_2);
    return 0;
}


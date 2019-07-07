# MeanFilterGPU
A gpu implementation of mean filter in CUDA

## Dependecies
In order to read, resize, and save the image files, [SOD - An Embedded Computer Vision & Machine Learning Library](https://sod.pixlab.io/) was used.

## Compilation

```
nvcc mean_filter.cu -o mean_filter -w
```

## Execution

```
./mean_filter <frame_size> <window_size>
```


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


//https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
//https://www.youtube.com/watch?v=DaEmuL0PYxc

__global__ void calculateHisto(char *buffer, int *histo, int numBins, int numElements)
{
    int threadNum = threadIdx.x + blockIdx.x * blockDim.x;
    int sectSize = numElements/blockDim.x;

    for (int k = 0; k < sectSize; k++) {
        buffer[threadNum+k]
        atomicAdd(&(histo[]), 1);
            }
        }
    }
}

int main()
{
    //Define and get size of input
    int numElements = 1000;
    int* input;
    input = (int*)malloc(sizeof(int) * numElements);

    //Populate
    for (int i = 0; i < numElements; i++) {
        input[i] = ceil(i/3);
        printf("%d ", input[i]);
    }
    printf("\n");

    //Define number of bins
    int numBins = 4;
    size_t binSize = numBins * sizeof(int);

    //Allocate histogram bins
    int* histo;
    histo = (int*)malloc(binSize);

    for (int i = 0; i < numBins; i++) {
        histo[i] = 0;
    }

    //Define and allocate deviceMemories
    char* deviceInput;
    int* deviceHisto;
    cudaMalloc((void**)&deviceInput, sizeof(input));
    cudaMalloc((void**)&deviceHisto, binSize);

    //Copy data to device
    cudaMemcpy(deviceInput, input, sizeof(input), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceHisto, histo, binSize, cudaMemcpyHostToDevice);

    int blockSize = 32;
    int gridSize = ceil(numElements / blockSize);

    calculateHisto <<< gridSize, blockSize >>> (deviceInput, deviceHisto, numBins, numElements);
    cudaDeviceSynchronize();

    cudaMemcpy(histo, deviceHisto, binSize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < numBins; i++) {
        printf("%d ", histo[i]);
    }
    printf("\n");

    cudaFree(deviceInput);
    cudaFree(deviceHisto);
    free(histo);

    return 0;
}
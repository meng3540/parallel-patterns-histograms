#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>

__global__ void calculateHisto(char* buffer, int* histo, int size, int numBins)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int sectSize = (size - 1) / (blockDim.x * gridDim.x) + 1;
    int start = i * sectSize;

    extern __shared__ unsigned int histos[];
    for (int binIdx = threadIdx.x; binIdx < numBins; binIdx += blockDim.x) {
        histos[binIdx] = 0u;
    }
    __syncthreads();

    for (int k = 0; k < sectSize; k++) {
        if (start + k < size) {
            int alphaPos = buffer[start + k] - 'a';
            if (alphaPos >= 0 && alphaPos < numBins) {
                atomicAdd(&(histos[alphaPos]), 1);
            }
        }
    }
    __syncthreads();
    for (int binIdx = threadIdx.x; binIdx < numBins; binIdx += blockDim.x) {
        atomicAdd(&(histo[binIdx]),histos[binIdx]);
    }
}

int main()
{
    FILE* file;
    char* buffer;
    long file_length;

    //Allow debugging of input by printing out input text. Very slow on large files so disabled by default.
    int previewInput = 0;

    // Open the file for reading
    file = fopen("enwik8", "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }

    // Get the length of the file
    fseek(file, 0, SEEK_END);
    file_length = ftell(file);
    rewind(file);

    // Allocate memory for the buffer to hold the file content
    buffer = (char*)malloc(file_length * sizeof(char));
    if (buffer == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        return 1;
    }

    // Read the file content into the buffer
    fread(buffer, file_length, 1, file);

    // Close the file
    fclose(file);
    
    //Define and get size of input
    char* input = buffer;

    if (previewInput) {
        printf("Input File:\n");
        for (int i = 0; i < file_length; ++i) {
            printf("%c", buffer[i]);
        }

        printf("\n\n");
    }
    //size_t inputSize = sizeof(input) - 1; // excluding null terminator
    
    //Define number of bins
    int numBins = 26; // number of alphabet letters
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
    cudaMalloc((void**)&deviceInput, file_length);
    cudaMalloc((void**)&deviceHisto, binSize);

    //Copy data to device
    cudaMemcpy(deviceInput, input, file_length, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceHisto, histo, binSize, cudaMemcpyHostToDevice);

    int blockSize = 32;
    int gridSize = ceil((float)file_length / blockSize); // adjust the gridSize calculation

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //Kernel Launch
    cudaEventRecord(start, 0);
    calculateHisto << < gridSize, blockSize >> > (deviceInput, deviceHisto, file_length, numBins);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(histo, deviceHisto, binSize, cudaMemcpyDeviceToHost);

    //Print histogram results
    printf("Histogram:\n");
    for (int i = 0; i < numBins; ++i) {
        printf("%c: %d\n", 'a' + i, histo[i]);
    }

    printf("\n");
    printf("The kernel took %.2f milliseconds to execute.\n", milliseconds);

    cudaFree(deviceInput);
    cudaFree(deviceHisto);
    free(histo);
    free(buffer);

    return 0;
}


# Histograms

#### Implementing and optimizing a CUDA-based histogram algorithm

### What is a Histogram?

A histogram is a graphical representation of the distribution of data. It displays the frequency or occurrence of values within specified intervals or bins. On a typical histogram, the horizontal axis is divided up into groups or bins (eg. a-d, e-h, i-l, etc.) and the number of occurrences of values in the data that are within a particular bin is graphed as a bar on the vertical axis. An example of a histogram is pictured below:

![Histogram of R](/res/Example_histogram.png "By Visnut - Using R from simulated data, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=36192473")

### Summary of Histograms and CUDA

As a basis for this assignment a summary of histograms, and how they may be accelerated by using NVIDIA's CUDA platform, was generated using Chat-GPT 3.5\. The output generated by Chat-GPT may be viewed in the [documentation in the docs folder](/doc/Histograms_Summary.md).

### Basic algorithm

The algorithm for the basic implementation of a histogram in CUDA was provided by the course textbook: [Programming Massively Parallel Processors, 3rd Edition by Morgan Kaufmann](https://learning.oreilly.com/library/view/programming-massively-parallel/9780128119877/).

First we make sure we have all the required libraries included:

```C++
__global__ void calculateHisto(char* buffer, int* histo, int size, int numBins)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int sectSize = (size - 1) / (blockDim.x * gridDim.x) + 1;
    int start = i * sectSize;

    for (int k = 0; k < sectSize; k++) {
        if (start + k < size) {
            int alphaPos = buffer[start + k] - 'a';
            if (alphaPos >= 0 && alphaPos < numBins) {
                atomicAdd(&(histo[alphaPos]), 1);
            }
        }
    }
}
```

The whole code can be found under the src folder.

In this implementation, the function takes in an array (the string of characters to take the histogram of), an output array that represents the histogram, a size parameter defining the size of the input array, and takes in the number of bins to sort into. **It should be noted that the basic algorithm does not account for binning multiple letters together, and only returns valid results for \26 bins; one per letter.**

The code defines some variables, namely i (the index for the current thread), sectSize (to split up the input array so each thread handles the right number of input elements so that all elements get processed) and a start variable (effectively the start offset in the input array for each thread to begin at).

```C++
int i = threadIdx.x + blockIdx.x * blockDim.x;
int sectSize = (size - 1) / (blockDim.x * gridDim.x) + 1;
int start = i * sectSize;
```

The function then iterates over the section assigned to the thread and if the current loop will still be processing input data (as in if the start position plus the current K value is still referring to an array index within the input array; there is valid work for this thread to do), then we calculate the numerical position of the character in the alphabet. This is achieved by subtracting "a" from the value in the input array. If the letter is a the result is 0, if the letter is b then the result is 1, and so on. If this value is a valid alphabet letter (from 0 to the number of bins), the count in the corresponding bin is incremented by \1. As the computations are happening in parallel, it is possible that two threads may try to increment the same bin at the same time. This would cause an issue (race condition) and therefore the atomicAdd() function is used to ensure only one thread can perform an action on an address (bin) at a time.

```C++
for (int k = 0; k < sectSize; k++) {
        if (start + k < size) {
            int alphaPos = buffer[start + k] - 'a';
            if (alphaPos >= 0 && alphaPos < numBins) {
                atomicAdd(&(histo[alphaPos]), 1);
            }
        }
    }
```

### Metrics: Timing Execution Time

To provide an easy metric to compare with, the code has an event timer added through the use of cudaEvent functions. The general form for adding these is as follows:

```C++
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

 //Kernel Launch
cudaEventRecord(start, 0);
kernel << < gridDim, blockDim>> > ();
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);

printf("\n");
printf("The kernel took %.2f milliseconds to execute.\n", milliseconds);
```

### Outputs

The results of running the non-optimized kernel are depicted below:

![Histogram of Lorem Ipsum](/res/NoOptim_Time.png "Kernel Runtime without optimization: 114ms")

### Optimizations

To optimize the 

###### Portions of this document were written using Chat-GPT. All information has been reviewed and deemed accurate by the authors.
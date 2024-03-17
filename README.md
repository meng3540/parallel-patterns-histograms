
# Histograms

#### Implementing and optimizing a CUDA-based histogram algorithm

### What is a Histogram?

A histogram is a graphical representation of the distribution of data. It displays the frequency or occurrence of values within specified intervals or bins. On a typical histogram, the horizontal axis is divided up into groups or bins (eg. a-d, e-h, i-l, etc.) and the number of occurrences of values in the data that are within a particular bin is graphed as a bar on the vertical axis. An example of a histogram is pictured below:

![Histogram of R](/res/Example_histogram.png "By Visnut - Using R from simulated data, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=36192473")

### Summary of Histograms and CUDA

AS a basis for this assignment a summary of histograms, and how they may be accelerated by using NVIDIA's CUDA platform, was generated using Chat-GPT \3.5. The output generated by Chat-GPT may be viewed in the [documentation in the docs folder](/doc/Histograms_Summary.rtf).

### Basic algorithm

The algorithm for the basic implementation of a histogram in CUDA was provided by the course textbook: [Programming Massively Parallel Processors, 3rd Edition by Morgan Kaufmann](https://learning.oreilly.com/library/view/programming-massively-parallel/9780128119877/).

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



###### Portions of this document were written using Chat-GPT. All information has been reviewed and deemed accurate by the authors.
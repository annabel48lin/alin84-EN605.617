#include <stdio.h> 
#include <vector>
#include <string>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define IMG_WIDTH 1024
#define IMG_HEIGHT 1024
#define CHANNELS 3 // R, G, B

__global__ void rgb_to_grayscale(
    unsigned char *rgb_input, 
    unsigned char *gray_output, 
    int img_width, 
    int img_height
) {
    int pixel_idx = threadIdx.x + blockIdx.x * blockDim.x ;
    int totalPixels = img_width * img_height;
    int stride = blockDim.x * gridDim.x;

    for (int idx = pixel_idx; idx < totalPixels; idx += stride) {
        int rgbIdx = idx * 3;
        unsigned char r = rgb_input[rgbIdx];
        unsigned char g = rgb_input[rgbIdx+1];
        unsigned char b = rgb_input[rgbIdx+2];

        gray_output[idx] = static_cast<unsigned char>(
            0.299f*r +0.587f*g + 0.114f*b);
    }
}

__global__ void edge_detection(
    unsigned char *gray_input, 
    unsigned char *edges_output, 
    int img_width, 
    int img_height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < img_width-1 && y >= 1 && y < img_height-1) {
        int idx_buffer = y * img_width + x;

        int gx = -gray_input[(y-1)*img_width + (x-1)] 
                - 2*gray_input[y*img_width + (x-1)] 
                - gray_input[(y+1)*img_width + (x-1)]
                + gray_input[(y-1)*img_width + (x+1)] 
                + 2*gray_input[y*img_width + (x+1)] 
                + gray_input[(y+1)*img_width + (x+1)];

        int gy = -gray_input[(y-1)*img_width + (x-1)] 
                - 2*gray_input[(y-1)*img_width + x] 
                - gray_input[(y-1)*img_width + (x+1)]
                + gray_input[(y+1)*img_width + (x-1)] 
                + 2*gray_input[(y+1)*img_width + x] 
                + gray_input[(y+1)*img_width + (x+1)];

        int mag = abs(gx) + abs(gy);
        edges_output[idx_buffer] = (unsigned char)((mag > 255) ? 255 : mag);
    }

}

// Write RGB image (width*height*3)
void writePNG_RGB(
    const std::string &filename, 
    const unsigned char *rgb, 
    int width, 
    int height
) {
    int success = stbi_write_png(filename.c_str(), width, height, 3, 
                                rgb, width*3);
    if (!success) {
        fprintf(stderr, "Failed to write PNG: %s\n", filename.c_str());
    }
}

// Write grayscale image (IMG_WIDTH*IMG_HEIGHT)
void writePNG_Gray(
    const std::string &filename, 
    const unsigned char *gray, 
    int width, 
    int height
) {
    int success = stbi_write_png(filename.c_str(), width, height, 1, 
                                    gray, width);
    if (!success) {
        fprintf(stderr, "Failed to write PNG: %s\n", filename.c_str());
    }
}

void do_image_processing(int totalThreads, int blockSize, int imgPattern) {
    // image sizes
    int imageSizeRGB = IMG_WIDTH * IMG_HEIGHT * CHANNELS;
    int imageSizeGray = IMG_WIDTH * IMG_HEIGHT;

    // Allocate host buffers
    unsigned char *h_rgb_input, *h_edges_result;
    cudaMallocHost((void**)&h_rgb_input, imageSizeRGB);
    cudaMallocHost((void**)&h_edges_result, imageSizeGray);

    // Populate image with RGB
    int patternSize = 32; // adjust to play with pattern
    switch (imgPattern) {
        case 1: // BW stripes
            for (int y = 0; y < IMG_HEIGHT; y++) {
                for (int x = 0; x < IMG_WIDTH; x++) {
                    int idx = (y*IMG_WIDTH + x)*3;
                    if ((x / patternSize) % 2 == 0) { 
                        h_rgb_input[idx]   = 255; // R
                        h_rgb_input[idx+1] = 255; // G
                        h_rgb_input[idx+2] = 255; // B
                    } else {
                        h_rgb_input[idx]   = 0;
                        h_rgb_input[idx+1] = 0;
                        h_rgb_input[idx+2] = 0;
                    }
                }
            }
            break;

        case 2: // BW checkerboard
            for (int y = 0; y < IMG_HEIGHT; y++) {
                for (int x = 0; x < IMG_WIDTH; x++) {
                    int idx = (y*IMG_WIDTH + x)*3;
                    bool white = ((x/patternSize) % 2) ^ ((y/patternSize) % 2);
                    unsigned char val = white ? 255 : 0;
                    h_rgb_input[idx]   = val; 
                    h_rgb_input[idx+1] = val; 
                    h_rgb_input[idx+2] = val; 
                }
            }
            break;
        
        case 3: // BW diagonal edge
            for (int y = 0; y < IMG_HEIGHT; y++) {
                for (int x = 0; x < IMG_WIDTH; x++) {
                    int idx = (y*IMG_WIDTH + x)*3;
                    if (x > y) {
                        h_rgb_input[idx]   = 255; 
                        h_rgb_input[idx+1] = 255; 
                        h_rgb_input[idx+2] = 255; 
                    } else {
                        h_rgb_input[idx]   = 0; 
                        h_rgb_input[idx+1] = 0; 
                        h_rgb_input[idx+2] = 0; 
                    }
                }
            }
            break;
        
        default: // random
            for (int i=0; i<imageSizeRGB; i++){
                h_rgb_input[i] = rand() % 256;
            }
            break;
    }

    // Allocate gpu memory
    unsigned char *d_rgb_input, *d_grayscale, *d_edges_result;
    cudaMalloc((void**)&d_rgb_input, imageSizeRGB);
    cudaMalloc((void**)&d_grayscale, imageSizeGray);
    cudaMalloc((void**)&d_edges_result, imageSizeGray);

    // Copy input to device
    cudaMemcpy(d_rgb_input, h_rgb_input, imageSizeRGB, cudaMemcpyHostToDevice);

    // Create events and streams
    cudaStream_t stream_grayscale, stream_edge;
    cudaStreamCreate(&stream_grayscale);
    cudaStreamCreate(&stream_edge);

    cudaEvent_t g_start, g_stop;
    cudaEvent_t e_start, e_stop;

    cudaEventCreate(&g_start);
    cudaEventCreate(&g_stop);
    cudaEventCreate(&e_start);
    cudaEventCreate(&e_stop);


    // launch RGB -> grayscale
    int numBlocks1D = (totalThreads + blockSize - 1) / blockSize;
    printf("Grayscale kernel 1D launch configuration:\n");
    printf("  User blockSize: %d threads per block\n", blockSize);
    printf("  Computed numBlocks: %d blocks\n", numBlocks1D);
    printf("  Total threads (numBlocks * blockSize): %d\n", numBlocks1D * blockSize);
    cudaEventRecord(g_start, stream_grayscale);
    rgb_to_grayscale<<<numBlocks1D, blockSize, 0, stream_grayscale>>>(d_rgb_input, d_grayscale, IMG_WIDTH, IMG_HEIGHT);
    cudaEventRecord(g_stop, stream_grayscale);

    // launch grayscale -> edge detection (wait for done_grayscale event)
    int sq_blockdim = static_cast<int>(std::sqrt(blockSize)); // e.g., 16
    dim3 block(sq_blockdim, sq_blockdim);
    // Grid dimensions (round up to cover full image)
    dim3 grid(
        (IMG_WIDTH  + block.x - 1) / block.x,
        (IMG_HEIGHT + block.y - 1) / block.y
    );
    printf("Edge Detection kernel 2D launch configuration:\n");
    printf("  User blockSize: %d\n", blockSize);
    printf("  Computed blockDim: (%d, %d) -> %d threads per block\n",
        block.x, block.y, block.x*block.y);
    printf("  Computed gridDim: (%d, %d) -> total threads (approx): %d\n",
       grid.x, grid.y, grid.x*grid.y*block.x*block.y);    

    cudaStreamWaitEvent(stream_edge, g_stop, 0);
    cudaEventRecord(e_start, stream_edge);
    edge_detection<<<grid, block, 0, stream_edge>>>(d_grayscale, d_edges_result, IMG_WIDTH, IMG_HEIGHT);
    cudaEventRecord(e_stop, stream_edge);

    cudaMemcpyAsync(h_edges_result, d_edges_result, imageSizeGray, cudaMemcpyDeviceToHost, stream_edge);

    // sync
    cudaEventSynchronize(e_stop);
    // cudaStreamSynchronize(stream_edge);

    // timings calc
    float msGray = 0.0f, msEdge = 0.0f, msTotal = 0.0f;
    cudaEventElapsedTime(&msGray, g_start, g_stop);
    cudaEventElapsedTime(&msEdge, e_start, e_stop);
    cudaEventElapsedTime(&msTotal, g_start, e_stop);

    printf("Grayscale kernel time: %.3f ms\n", msGray);
    printf("Edge kernel time:      %.3f ms\n", msEdge);
    printf("Total pipeline time:   %.3f ms\n", msTotal);

    // write images
    writePNG_RGB("rgb.png", h_rgb_input, IMG_WIDTH, IMG_HEIGHT);
    writePNG_Gray("edges.png", h_edges_result, IMG_WIDTH, IMG_HEIGHT);
    
    cudaFree(d_rgb_input);
    cudaFree(d_grayscale);
    cudaFree(d_edges_result);
    cudaFreeHost(h_rgb_input);
    cudaFreeHost(h_edges_result);
}

int main(int argc, char* argv[]) {

    // read comand line arguments
    int totalThreads = (1<<20);
    int blockSize = 256;
    int imgPattern = 0;

    if (argc >= 2) {
        totalThreads = atoi(argv[1]);
    }
    if (argc >= 3) {
        blockSize = atoi(argv[2]);
    }
    if (argc >= 4) {
        imgPattern = atoi(argv[3]);
    }

    printf("Run w totalThreads %d, blockSize %d\n", 
        totalThreads, blockSize);

    do_image_processing(totalThreads, blockSize, imgPattern);
}
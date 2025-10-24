#include <iostream>
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include <npp.h>
#include <nppcore.h> 
#include <nppi.h>
#include <nppdefs.h>

#include <filesystem>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void threshold_kernel(
    unsigned char* d_input, 
    unsigned char* d_output, 
    int width, 
    int height, 
    unsigned char threshold
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        d_output[idx] = (d_input[idx] > threshold) ? 0 : 255;
    }
}

__device__ inline int isPixelBlack(
    const unsigned char* img , 
    int x, 
    int y, 
    int w, 
    int h
) {
    if (x < 0 || x >= w || y<0 || y>=h) return 0;
    return img[y*w+x] > 0 ? 1 : 0;
}

__global__
void ZhangSuenMark(
    const unsigned char* src, 
    unsigned char* mark, 
    int w, 
    int h, 
    int pass
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int p1 = isPixelBlack(src, x, y, w, h);
    // ! the pixel is black (cond0)
    if (!p1) {
        mark[y*w + x] = 0; 
        return;
    }

    int p2 = isPixelBlack(src, x-1, y-1, w, h); // top left
    int p3 = isPixelBlack(src, x-1, y, w, h);   // left
    int p4 = isPixelBlack(src, x-1, y+1, w, h); // bottom left
    int p5 = isPixelBlack(src, x, y+1, w, h);   // bottom
    int p6 = isPixelBlack(src, x+1, y+1, w, h); // bottom right
    int p7 = isPixelBlack(src, x+1, y, w, h);   // right 
    int p8 = isPixelBlack(src, x+1, y-1, w, h); // top right
    int p9 = isPixelBlack(src, x, y-1, w, h);   // top


    // A(i,j) = number of transitions from white to black 
    //          in sequence of 8 neighbors making complete circle
    int A = 0;
    int sequence[8] = {p2, p3, p4, p5, p6, p7, p8, p9};
    for (int i=0; i<8; i++) {
        if (sequence[i] == 0 && sequence[(i+1)%8]==1) {
            A++; // white --> black
        }
    }

    // B(i,j) = number of black pixels among 8 neighbors
    int B = p2+p3+p4+p5+p6+p7+p8+p9;

    // 1. 2 <= B(i,j)
    bool cond1 = (B>=2 && B<=6);
    // 2. A(i,j) = 1
    bool cond2 = (A==1); 
    
    // 3.
    bool cond3, cond4; 
    if (pass == 0) { 
        // 3. at least one of north, east, south neighbors is white
        cond3 = (!p9 || !p7 || !p5);
        // 4. at least one of east, south, west neightbors is white
        cond4 = (!p7 || !p5 || !p3);
    } else {
        // 3. at least one of north, east, west neigbors is white
        cond3 = (!p9 || !p7 || !p3);
        // 4. at least one of north, south, west neighbors is white
        cond4 = (!p9 || !p5 || !p3);
    }

    if (cond1 && cond2 && cond3 && cond4) {
        mark[y*w + x] = 1; // mark to remove
    }
    else {
        mark[y*w + x] = 0;
    }
}

__global__ void removeMarked(unsigned char* img, const unsigned char* mark, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    if (mark[y*w + x]) img[y*w + x] = 0;
}

void setupContext(NppStreamContext& ctx) {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    ctx.nCudaDeviceId = device;
    ctx.nMultiProcessorCount = prop.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock = prop.sharedMemPerBlock;
    ctx.nCudaDevAttrComputeCapabilityMajor = prop.major;
    ctx.nCudaDevAttrComputeCapabilityMinor = prop.minor;
}

int main(int argc , char** argv) {
    if (argc < 2) {
        std::cerr << "Args: -i imagePath" << std::endl;
        return 1;
    }

    int w, h, channels;
    unsigned char* h_img_rgb = stbi_load(argv[1], &w, &h, &channels, 3);
    if (!h_img_rgb) {
        std::cerr << "Failed to load image\n";
        return 1;
    }

    unsigned char *d_img_rgb;
    cudaMalloc(&d_img_rgb, w*h*3);
    cudaMemcpy(d_img_rgb, h_img_rgb, w*h*3, cudaMemcpyHostToDevice);

    NppStreamContext ctx = {};
    setupContext(ctx);

    // NPP: RGB --> grayscale
    NppStatus status;
    unsigned char* d_gray;
    cudaMalloc(&d_gray, w*h);
    NppiSize npp_dim = {w, h};
    status = nppiRGBToGray_8u_C3C1R_Ctx(d_img_rgb, w*3, 
                                        d_gray, w, npp_dim, ctx);
    if (status != NPP_SUCCESS) {
        std::cerr << "Failed to convert image to grayscale.\n";
        return 1;
    }

    // grayscale --> B/W
    unsigned char* d_bw;
    cudaMalloc(&d_bw, w*h);
    
    int BLOCK = 16;
    dim3 threads(BLOCK,BLOCK);
    dim3 blocks((w+BLOCK-1)/BLOCK,(h+BLOCK-1)/BLOCK);

    threshold_kernel<<<blocks,threads>>>(d_gray, d_bw, w, h, 128);
    cudaDeviceSynchronize();

    // skeletonization (Zhang Suen) 
    unsigned char* d_mark;
    cudaMalloc(&d_mark, w*h);

    bool changed;
    do {
        changed = false;
        for (int pass=0; pass<2; pass++) { // pass 1, 2
            ZhangSuenMark<<<blocks,threads>>>(d_bw, d_mark, w, h, pass);
            cudaDeviceSynchronize();

            // Thrust: count marked pixels
            thrust::device_ptr<unsigned char> dev_mark(d_mark);
            int numMarked = thrust::reduce(thrust::device, 
                                            dev_mark, dev_mark + w*h, 0);
            if(numMarked > 0) changed = true;

            removeMarked<<<blocks,threads>>>(d_bw, d_mark, w, h);
            cudaDeviceSynchronize();
        }
    } while(changed);

    // Copy result back
    unsigned char* h_result = new unsigned char[w*h];
    cudaMemcpy(h_result, d_bw, w*h, cudaMemcpyDeviceToHost);

    // write to file
    std::filesystem::path input_path(argv[1]);
    std::string base_name = input_path.stem().string();
    std::filesystem::path output_path = std::filesystem::path("outputs") / 
                                            (base_name + "_skeleton.png");
    std::filesystem::create_directories("outputs");
    std::string output_filename = output_path.string();
    std::cout << "Writing to " << output_filename << std::endl;
    stbi_write_png(output_filename.c_str(), w, h, 1, h_result, w);

    // Cleanup
    cudaFree(d_img_rgb);
    cudaFree(d_gray);
    cudaFree(d_bw);
    cudaFree(d_mark);
    delete[] h_result;

    return 0;
}
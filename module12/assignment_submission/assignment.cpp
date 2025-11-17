// This code is based off of the given simple.cpp program, and adapted to use for my assignment\ 

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>

#include "info.hpp"

#define DEFAULT_PLATFORM 0

// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) 
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_uint numDevices;
    cl_platform_id * platformIDs;
    cl_device_id * deviceIDs;
    cl_context context;
    cl_program program;
    std::vector<cl_kernel> kernels;
    std::vector<cl_command_queue> queues;
    std::vector<cl_mem> buffers;
    int * inputOutput;

    int platform = DEFAULT_PLATFORM; 

    // sim arg params
    int width = 128;
    int height = 128;
    int secs = 10;
    float a = 10.0f;     // diffusion constant

    std::cout << "2D Heat Diffusion Simulation" << std::endl;

    for (int i = 1; i < argc; i++)
    {
        std::string input(argv[i]);

        if (!input.compare("--platform"))
        {
            input = std::string(argv[++i]);
            std::istringstream buffer(input);
            buffer >> platform;
        }
        else if (!input.compare("--width") && argc >= i+1) {
            width = std::stoi(argv[++i]);
        }
        else if (!input.compare("--height") && argc >= i+1) {
            height = std::stoi(argv[++i]);
        }
        else if (!input.compare("--secs") && argc >= i+1) {
            secs = std::stoi(argv[++i]);
        }
        else if (!input.compare("--a") && argc >= i+1) {
            a = std::stof(argv[++i]);
        }
        else
        {
            std::cout << "usage (all optional): --platform n --width w "
            << "--height h --secs s --a diffusion_constant" << std::endl;
            return 0;
        }
    }

    std::cout << "width " << width << ", height " << height << 
        ", secs " << secs << ", a " << a << std::endl;

    // non-input sim params
    float dt = .01f; // steps per second
    float aTimesDt = a * dt;
    int steps = static_cast<int>(secs / dt);

    
    // First, select an OpenCL platform to run on.  
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr( 
        (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
        "clGetPlatformIDs"); 
 
    platformIDs = (cl_platform_id *)alloca(
            sizeof(cl_platform_id) * numPlatforms);

    std::cout << "Number of platforms: \t" << numPlatforms << std::endl; 

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
       (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
       "clGetPlatformIDs");

    std::ifstream srcFile("heat_diffusion.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading heat_diffusion.cl");

    std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

    const char * src = srcProg.c_str();
    size_t length = srcProg.length();

    deviceIDs = NULL;
    DisplayPlatformInfo(
        platformIDs[platform], 
        CL_PLATFORM_VENDOR, 
        "CL_PLATFORM_VENDOR");

    errNum = clGetDeviceIDs(
        platformIDs[platform], 
        CL_DEVICE_TYPE_ALL, 
        0,
        NULL,
        &numDevices);
    if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
    {
        checkErr(errNum, "clGetDeviceIDs");
    }       

    deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
    errNum = clGetDeviceIDs(
        platformIDs[platform],
        CL_DEVICE_TYPE_ALL,
        numDevices, 
        &deviceIDs[0], 
        NULL);
    checkErr(errNum, "clGetDeviceIDs");

    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[platform],
        0
    };

    context = clCreateContext(
        contextProperties, 
        numDevices,
        deviceIDs, 
        NULL,
        NULL, 
        &errNum);
    checkErr(errNum, "clCreateContext");

    // Create program from source
    program = clCreateProgramWithSource(
        context, 
        1, 
        &src, 
        &length, 
        &errNum);
    checkErr(errNum, "clCreateProgramWithSource");

    // Build program
    errNum = clBuildProgram(
        program,
        numDevices,
        deviceIDs,
        "-I.",
        NULL,
        NULL);
    if (errNum != CL_SUCCESS) 
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(
            program, 
            deviceIDs[0], 
            CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog), 
            buildLog, 
            NULL);

            std::cerr << "Error in OpenCL C source: " << std::endl;
            std::cerr << buildLog;
            checkErr(errNum, "clBuildProgram");
    }
    // create kernel
    cl_kernel kernel = clCreateKernel(program, "diffuse_step", &errNum);
    checkErr(errNum, "clCreateKernel");

    // heat grid
    int gridSize = width * height;
    std::vector<float> grid(gridSize, 0.0f);
    srand(time(NULL));
    grid[(height/2)*width + width/2] = 200.0f; // heat source at center

    // main buffer
    cl_mem main_buffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * gridSize,
        grid.data(),
        &errNum);
    checkErr(errNum, "clCreateBuffer");

    // sub-buffers per device
    std::vector<size_t> rowOffsets(numDevices);
    std::vector<size_t> rowsPerDevice(numDevices);

    size_t baseRows = height / numDevices;
    size_t remainder = height % numDevices;

    size_t offset = 0;
    for (unsigned d = 0; d < numDevices; d++) {
        size_t rows = baseRows + (d < remainder ? 1 : 0); // distribute rows
        rowOffsets[d] = offset;
        rowsPerDevice[d] = rows;
        offset += rows;
    }

    std::vector<cl_mem> subBuffers;
    for (unsigned d = 0; d < numDevices; d++) {
        cl_buffer_region region = { rowOffsets[d] * width * sizeof(float), 
            rowsPerDevice[d] * width * sizeof(float) };
        cl_mem sub = clCreateSubBuffer(main_buffer, CL_MEM_READ_WRITE, 
            CL_BUFFER_CREATE_TYPE_REGION, &region, &errNum);
        checkErr(errNum, "clCreateSubBuffer");
        subBuffers.push_back(sub);
    }

    // Create command queues
    for (unsigned int i = 0; i < numDevices; i++)
    {
        InfoDevice<cl_device_type>::display(
            deviceIDs[i], 
            CL_DEVICE_TYPE, 
            "CL_DEVICE_TYPE");

        cl_command_queue queue = 
            clCreateCommandQueue(
                context,
                deviceIDs[i],
                0,
                &errNum);
        checkErr(errNum, "clCreateCommandQueue");

        queues.push_back(queue);
    }

    // ffmpeg output
    int frameSkip = 10;
    int fps = 1.0 / (dt*frameSkip); 
    FILE* ffmpeg = _popen(
        ("ffmpeg -y -f rawvideo -pix_fmt rgb24 -s " +
        std::to_string(width) + "x" + std::to_string(height) +
        " -r " + std::to_string(fps) +
        " -i - "                  
        "-c:v libx264 "           
        "-pix_fmt yuv420p "           
        "heat2d.mp4").c_str(),
        "wb"
    );
    checkErr(ffmpeg ? CL_SUCCESS : -1, "opening ffmpeg pipe");
                          
    // simulation loop
    std::vector<float> nextGrid(gridSize, 0.0f);
    cl_mem nextBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 
        sizeof(float)*gridSize, nullptr, &errNum);
    checkErr(errNum, "clCreateBuffer next");
    for (int s = 0; s < steps; s++) {
        // Launch kernel per sub-buffer
        for (unsigned d = 0; d < numDevices; d++) {
            errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &subBuffers[d]);
            errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &nextBuffer);
            errNum |= clSetKernelArg(kernel, 2, sizeof(int), &width);
            errNum |= clSetKernelArg(kernel, 3, sizeof(int), &height);
            errNum |= clSetKernelArg(kernel, 4, sizeof(int), &rowOffsets[d]);
            errNum |= clSetKernelArg(kernel, 5, sizeof(float), &aTimesDt);

            checkErr(errNum, "clSetKernelArg");

            size_t global[2] = { width, rowsPerDevice[d] };
            clEnqueueNDRangeKernel(queues[d], kernel, 2, nullptr, global, 
                nullptr, 0, nullptr, nullptr);
        }

        clFinish(queues[0]); // wait for all

        // Read back for output frame
        clEnqueueReadBuffer(queues[0], nextBuffer, CL_TRUE, 0, 
            sizeof(float)*gridSize, nextGrid.data(), 0, nullptr, nullptr);

        // Write frame to FFmpeg
        if (s % frameSkip == 0) {
            std::vector<unsigned char> img(gridSize*3);
            for (size_t i = 0; i < gridSize; i++) {
                unsigned char r = static_cast<unsigned char>(std::min(
                    255.0f, nextGrid[i] * 255.0f));
                img[3 * i + 0] = r; // R
                img[3 * i + 1] = 0; // G
                img[3 * i + 2] = 0; // B
            }
            fwrite(img.data(), sizeof(unsigned char), gridSize * 3, ffmpeg);
        }
        // swap buffers so we can use nextBuffer as input
        std::swap(main_buffer, nextBuffer); 
    }

    _pclose(ffmpeg);

    std::cout << "Program completed successfully" << std::endl;

    return 0;

}
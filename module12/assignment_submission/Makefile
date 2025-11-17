OPENCL_INC = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\include
OPENCL_LIB = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\lib\x64

CC = cl
CFLAGS = /I"$(OPENCL_INC)"
LDFLAGS = /link /LIBPATH:"$(OPENCL_LIB)" OpenCL.lib
all: assignment.cpp
	cl assignment.cpp $(CFLAGS) $(LDFLAGS) /OUT:assignment.exe

run:
	assignment.exe
	
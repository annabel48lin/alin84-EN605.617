INCLUDES = -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\include"
LIBDIRS = -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\lib\x64"

NPPLIBS = -lnppc -lnppicc -lnppist

EXE := assignment.exe

INPUT_DIR := inputs
INPUT_FILES := $(wildcard $(INPUT_DIR)/*)


all: assignment.cu
	nvcc -std=c++17 $(INCLUDES) assignment.cu -o $(EXE) $(LIBDIRS) $(NPPLIBS)


run_all:
	@echo Running $(EXE) on each file...
	@for %%f in ($(INPUT_FILES)) do ( \
		echo Processing %%f... && \
		$(EXE) "%%f" \
	)

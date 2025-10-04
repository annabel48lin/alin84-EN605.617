Text and images in LinAnnabel_Module6Assignment_Writeup.pdf

GPU code in assignment.cu

To compile: 
using nvcc: nvcc assignment.cu -o assignment.exe
using Makefile: make 

To run: 
assignment.exe totalThreads blockSize imgPattern

imgPattern 
0 = random (default if no match)
1 = vertical stripes
2 = checkerboard
3 = diagonal edge

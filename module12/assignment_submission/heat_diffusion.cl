__kernel void diffuse_step(
    __global float* map_input, 
    __global float* map_output, 
    int width,
    int height,
    int row_offset, // of sub-buffer
    float a)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    i = i + row_offset; // to account for the sub-buffers

    if (i >= width || j >= height) return;

    int idx = i * width + j;

    float uij = map_input[idx];
    float uim1j = (i > 0) ? map_input[(i-1)*width + j] : 0.;
    float uip1j = (i < width-1) ? map_input[(i+1)*width + j] : 0.;
    float uijm1 = (j > 0) ? map_input[i*width + (j-1)] : 0.;
    float uijp1 = (j < height-1) ? map_input[i*width + (j+1)] : 0.;

    map_output[idx] = uij + a * ((uim1j - 2.0f*uij + uip1j) + (uijm1 - 2.0f*uij + uijp1));
}
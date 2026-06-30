__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int N = 1 << 20;
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    
    float h_a[1024], h_b[1024], h_c[1024];
    for (int i = 0; i < 1024; i++) { h_a[i] = (float)i / 1024.0f; h_b[i] = 1.0f - h_a[i]; }
    
    cudaMemcpy(d_a, h_a, 1024*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, 1024*sizeof(float), cudaMemcpyHostToDevice);
    vectorAdd<<<4, 256>>>(d_a, d_b, d_c, 1024);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c, d_c, 1024*sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1;
    
    int ok = 1;
    for (int i = 0; i < 1024; i++) {
        float diff = h_c[i] - (h_a[i] + h_b[i]);
        if (diff < 0) diff = -diff;
        if (diff > 1e-5f) { ok = 0; break; }
    }
    
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return ok ? 0 : 1;
}

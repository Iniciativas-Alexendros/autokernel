#include <cstdio>

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
    float *h_a = new float[N], *h_b = new float[N], *h_c = new float[N];
    for (int i = 0; i < N; i++) { h_a[i] = float(rand()) / RAND_MAX; h_b[i] = float(rand()) / RAND_MAX; }
    cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice);
    vectorAdd<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);
    bool ok = true;
    for (int i = 0; i < N; i++) if (fabs(h_c[i]-h_a[i]-h_b[i]) > 1e-5f) { ok = false; break; }
    printf("Vector addition %s (exit %d)\n", ok ? "PASSED" : "FAILED", ok ? 0 : 1);
    delete[] h_a; delete[] h_b; delete[] h_c;
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return ok ? 0 : 1;
}

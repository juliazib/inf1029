#include "matrix_lib.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void scalar_mult_kernel(float scalar, float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = scalar * input[idx];
    }
}

int scalar_matrix_mult(float scalar_value, matrix *m, matrix *r) {
    if (!m || !r || !m->values || !r->values)
        return -2;

    int size = m->rows * m->cols;
    float *d_input = NULL, *d_output = NULL;

    cudaMalloc((void **)&d_input, size * sizeof(float));
    cudaMalloc((void **)&d_output, size * sizeof(float));

    cudaMemcpy(d_input, m->values, size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = threadsPerBlock;
    int blocks = (size + threads - 1) / threads;

    scalar_mult_kernel<<<blocks, threads>>>(scalar_value, d_input, d_output, size);
    cudaDeviceSynchronize();
    
    cudaMemcpy(r->values, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

__global__ void matrix_mult_kernel(float *mA, float *mB, float *mC, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < m && col < p) {
        float sum = 0.0f;

        // ponteiros para linha de A e coluna de B
        float *a_row = mA + row * n;
        float *b_col = mB + col;

        for (int j = 0; j < n; ++j) {
            float a = *(a_row + j);
            float b = *(b_col + j * p);
            sum += a * b;
        }
        *(mC + row * p + col) = sum;
    }
}

__global__ void matrix_matrix_linear_kernel(float* m1, float* m2, float* r, int m, int n, int p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    float* m1_ptr = m1 + i * n;
    float* r_ptr = r + i * p;

    float* m1_elem_ptr = m1_ptr;
    float* r_row_ptr = r_ptr;

    for(int j = 0; j < n; j++) {
        float value = *m1_elem_ptr;
        m1_elem_ptr++;

        float* m2_row_ptr = m2 + j * p;
        float* r_col_ptr = r_row_ptr;
        
        int k = 0;

        while(k < p) {
            *r_col_ptr += value * m2_row_ptr;
            r_col_ptr++;
            m2_row_ptr++;
            k++;
        }
    }
}


int matrix_matrix_mult(matrix *m1, matrix *m2, matrix *r) {
    if (!m1 || !m2 || !r || !m1->values || !m2->values || !r->values)
        return -2;

    if (m1->cols != m2->rows)
        return -1;

    int m = m1->rows;
    int n = m1->cols;
    int p = m2->cols;

    r->rows = m;
    r->cols = p;

    memset(r->values, 0, sizeof(float) * m * p);

    size_t sizeA = m * n * sizeof(float);
    size_t sizeB = n * p * sizeof(float);
    size_t sizeC = m * p * sizeof(float);

    float *d_A = NULL, *d_B = NULL, *d_C = NULL;

    cudaError_t err;

    err = cudaMalloc((void **)&d_A, sizeA);
    if (err != cudaSuccess) return -3;

    err = cudaMalloc((void **)&d_B, sizeB);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        return -3;
    }

    err = cudaMalloc((void **)&d_C, sizeC);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        return -3;
    }

    err = cudaMemcpy(d_A, m1->values, sizeA, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -3;
    }

    err = cudaMemcpy(d_B, m2->values, sizeB, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -3;
    }

    int threads_x = 16;                           
    int threads_y = threadsPerBlock / threads_x;  
    int blocks_x = (p + threads_x - 1) / threads_x;
    int blocks_y = (m + threads_y - 1) / threads_y;

    dim3 threads(threads_x, threads_y);
    dim3 blocks(blocks_x, blocks_y);

    matrix_mult_kernel<<<blocks, threads>>>(d_A, d_B, d_C, m, n, p);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -3;
    }
    

    err = cudaMemcpy(r->values, d_C, sizeC, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -3;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

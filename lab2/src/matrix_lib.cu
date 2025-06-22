#include "matrix_lib.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void scalar_mult_kernel(float scalar, float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float *in_ptr = input + idx;
        float *out_ptr = output + idx;
        *out_ptr = scalar * (*in_ptr);
    }
}

int scalar_matrix_mult(float scalar_value, matrix *m, matrix *r) {
    if (!m || !r || !m->values || !r->values)
        return -2;

    int size = m->rows * m->cols;
    float *cinput = NULL, *coutput = NULL;

    cudaError_t err;

    err = cudaMalloc((void **)&cinput, size * sizeof(float));
    if (err != cudaSuccess) return -3;

    err = cudaMalloc((void **)&coutput, size * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(input);
        return -3;
    }

    err = cudaMemcpy(cinput, m->values, size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(cinput);
        cudaFree(coutput);
        return -3;
    }

    int threads = threadsPerBlock;
    int blocks = (size + threads - 1) / threads;

    scalar_mult_kernel<<<blocks, threads>>>(scalar_value, cinput, coutput, size);
    cudaDeviceSynchronize();
    
    err = cudaMemcpy(r->values, coutput, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    if (err != cudaSuccess) {
        cudaFree(cinput);
        cudaFree(coutput);
        return -3;
    }

    cudaFree(cinput);
    cudaFree(coutput);

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

    float* cA = NULL; 
    float* cB = NULL; 
    float* cC = NULL;

    cudaError_t err;

    err = cudaMalloc((void **)&cA, sizeA);
    if (err != cudaSuccess) return -3;

    err = cudaMalloc((void **)&cB, sizeB);
    if (err != cudaSuccess) {
        cudaFree(cA);
        return -3;
    }

    err = cudaMalloc((void **)&cC, sizeC);
    if (err != cudaSuccess) {
        cudaFree(cA);
        cudaFree(cB);
        return -3;
    }

    err = cudaMemcpy(cA, m1->values, sizeA, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(cA);
        cudaFree(cB);
        cudaFree(cC);
        return -3;
    }

    err = cudaMemcpy(cB, m2->values, sizeB, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(cA);
        cudaFree(cB);
        cudaFree(cC);
        return -3;
    }

    int threads_x = 16;                           
    int threads_y = threadsPerBlock / threads_x;  
    int blocks_x = (p + threads_x - 1) / threads_x;
    int blocks_y = (m + threads_y - 1) / threads_y;

    dim3 threads(threads_x, threads_y);
    dim3 blocks(blocks_x, blocks_y);

    matrix_mult_kernel<<<blocks, threads>>>(cA, cB, cC, m, n, p);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(cA);
        cudaFree(cB);
        cudaFree(cC);
        return -3;
    }
    

    err = cudaMemcpy(r->values, cC, sizeC, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(cA);
        cudaFree(cB);
        cudaFree(cC);
        return -3;
    }

    cudaFree(cA);
    cudaFree(cB);
    cudaFree(cC);

    return 0;
}

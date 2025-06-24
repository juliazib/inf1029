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
	if (!m || !r || !m->values || !r->values) return -1;
    if(m->rows != r->rows || m->cols != r->cols) return -2;

	int size = m->rows * m->cols;
	float *deviceInput = NULL, *deviceOutput = NULL;

	cudaError_t err;
	err = cudaMalloc((void**)&deviceInput, size * sizeof(float));
	if (err != cudaSuccess){
        return -3;
    }
	err = cudaMalloc((void**)&deviceOutput, size * sizeof(float));
	if (err != cudaSuccess) {
		cudaFree(deviceInput);
		return -3;
	}

	err = cudaMemcpy(deviceInput, m->values, size * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cudaFree(deviceInput); cudaFree(deviceOutput);
		return -4;
	}

    int max_blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    int blocks = (blocksPerGrid >= max_blocks) ? blocksPerGrid : max_blocks;

    //printf("Kernel config: %d blocks x %d threads = %d threads, size = %d\n", blocks, threadsPerBlock, blocks * threadsPerBlock, size);

	scalar_mult_kernel<<<blocks, threadsPerBlock>>>(scalar_value, deviceInput, deviceOutput, size);
    cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess) {
        fprintf(stderr, "Erro no kernel: %s\n", cudaGetErrorString(err));
		cudaFree(deviceInput); cudaFree(deviceOutput);
		return -5;
	}

	err = cudaMemcpy(r->values, deviceOutput, size * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		cudaFree(deviceInput); cudaFree(deviceOutput);
		return -4;
	}

	cudaFree(deviceInput);
	cudaFree(deviceOutput);

	return 0;
}

__global__ void matrix_mult_1d(float *mA, float *mB, float *mC, int m, int n, int p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * p) return;

    int row = idx / p;
    int col = idx % p;

    float sum = 0.0f;

    float* m1_ptr = mA + row * n;
    float* m2_ptr = mB + col;
    
    for (int k = 0; k < n; k++) {
        sum += *(m1_ptr + k) * *(m2_ptr + k * p);
    }

    mC[idx] = sum;
}

int matrix_matrix_mult(matrix *m1, matrix *m2, matrix *r) {
    if (!m1 || !m2 || !r || !m1->values || !m2->values || !r->values)
        return -1;

    if (m1->cols != m2->rows)
        return -2;

    int m = m1->rows;
    int n = m1->cols;
    int p = m2->cols;


    memset(r->values, 0, sizeof(float) * m * p);

    size_t sizeA = m * n * sizeof(float);
    size_t sizeB = n * p * sizeof(float);
    size_t sizeC = m * p * sizeof(float);

    float* deviceA = NULL; 
    float* deviceB = NULL; 
    float* deviceC = NULL;

    cudaError_t err;

    err = cudaMalloc((void **)&deviceA, sizeA);
    if (err != cudaSuccess) return -3;

    err = cudaMalloc((void **)&deviceB, sizeB);
    if (err != cudaSuccess) {
        cudaFree(deviceA);
        return -3;
    }

    err = cudaMalloc((void **)&deviceC, sizeC);
    if (err != cudaSuccess) {
        cudaFree(deviceA);
        cudaFree(deviceB);
        return -3;
    }

    err = cudaMemcpy(deviceA, m1->values, sizeA, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceC);
        return -4;
    }

    err = cudaMemcpy(deviceB, m2->values, sizeB, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceC);
        return -4;
    }

    int size = m * p;
    int max_blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    int blocks = (blocksPerGrid >= max_blocks) ? blocksPerGrid : max_blocks;
    //printf("Kernel config: %d blocks x %d threads = %d threads, size = %d\n", blocks, threadsPerBlock, blocks * threadsPerBlock, size);

    matrix_mult_1d<<<blocks, threadsPerBlock>>>(deviceA, deviceB, deviceC, m, n, p);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    
    if (err != cudaSuccess) {
      fprintf(stderr, "Erro no kernel: %s\n", cudaGetErrorString(err));
      cudaFree(deviceA); 
      cudaFree(deviceB); 
      cudaFree(deviceC);
      return -5;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceC);
        return -5;
    }
    

    err = cudaMemcpy(r->values, deviceC, sizeC, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceC);
        return -4;
    }

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return 0;
}

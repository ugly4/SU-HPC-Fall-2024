#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <chrono>

const int N = 2000;

__global__ void gpu_matrix_mult(const int* A, const int* B, int* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
	
    if (row < n && col < n)
	{
        int sum = 0;
        for (int i = 0; i < n; ++i) 
		{
            sum += A[row * n + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

void cpu_matrix_mult(const int *h_a, const int *h_b, int *h_result, int n) {
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < n; ++j) 
        {
            int tmp = 0.0;
            for (int k = 0; k < n; ++k) 
            {
                tmp += h_a[i * n + k] * h_b[k * n + j];
            }
            h_result[i * n + j] = tmp;
        }
    }
}

void measureTime(void(*func)(const int*, const int*, int*, int), const int* A, const int* B, int* C, int n, const char* description) 
{
    auto start = std::chrono::high_resolution_clock::now();
    func(C, A, B, n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("%s took %f seconds\n", description, diff.count());
}

bool compareMatrices(const int* A, const int* B, int n) {
    for (int i = 0; i < n * n; ++i) 
	{
        if (A[i] != B[i]) 
		{
            return false;
        }
    }
	
    return true;
}

int main() 
{
    int* A = new int[N * N];
    int* B = new int[N * N];
    int* C = new int[N * N];
    int* C_CPU = new int[N * N];
	
    for (int i = 0; i < N * N; i++) 
	{
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }
	
    measureTime(matrixMulCUDA, A, B, C, N, "CUDA");
    measureTime(matrixMulCPU, A, B, C_CPU, N, "CPU");
	
    if (compareMatrices(C, C_CPU, N)) 
	{
        printf("Results are correct!\n");
    }	else 
	{
        printf("Results are incorrect!\n");
    }
	
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_CPU;
	
    return 0;
}
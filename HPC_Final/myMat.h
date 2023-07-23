#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <mpi.h>
#include <cstdlib>
using namespace std;

class MyMat {
private:
    /*
        随机矩阵生成器
        1. 输入行列数
    */
    int** getMat(int row, int col);

    /*
        获取稀疏矩阵
        1. 输入行数
        2. 输入列数
        3. 输入稀疏度
    */
    int** getSparseMatrix(int row_size, int col_size, double sparsity);

    void matMultiplyWithSingleThread(int* A, int* B, int* matResult, int m, int p, int n);

public:
    /*
        打印矩阵
    */
    void printMatrix(int** matrix, int row_size, int col_size);

    /*
        矩阵A生成器
        1. 输入行数（列数必须为11），输出随机矩阵
        2. 矩阵的第一行为学号
    */
    int** getMatA(int row);

    /*
        矩阵B生成器
        1. 输入列数（行数必须为11，对应矩阵A），输出随机矩阵
    */
    int** getMatB(int col);

    /*
        稀疏矩阵A生成器
        1. 输入行数（列数必须为11）
        2. 输入稀疏度
    */
    int** getSparseMatA(int row_size, double sparsity);

    /*
        稀疏矩阵B生成器
        1. 输入列数（行数必须为11）
        2. 输入稀疏度
    */
    int** getSparseMatB(int col_size, double sparsity);

    /*
        对矩阵进行预处理
        1. 对于[x0-average(x1,…,x8)]/ average(x1,…,x8)  <  10%，成立则不改变M(x0, y0)的值，反之则用均值覆盖
    */
    void meanFilter(int** mat, int row_size, int col_size);

    /*
        通用矩阵乘法
        1. 输入矩阵A
        2. 输入矩阵B
    */
    int** product(int** matA, int rowNumA, int** matB, int colNumB);

    /*
        稀疏矩阵乘法
        1. 输入矩阵A
        2. 输入矩阵B
    */
    int** sparseProduct(int** matA, int rowNumA, int** matB, int colNumB);

};
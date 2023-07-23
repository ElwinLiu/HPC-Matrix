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
        �������������
        1. ����������
    */
    int** getMat(int row, int col);

    /*
        ��ȡϡ�����
        1. ��������
        2. ��������
        3. ����ϡ���
    */
    int** getSparseMatrix(int row_size, int col_size, double sparsity);

    void matMultiplyWithSingleThread(int* A, int* B, int* matResult, int m, int p, int n);

public:
    /*
        ��ӡ����
    */
    void printMatrix(int** matrix, int row_size, int col_size);

    /*
        ����A������
        1. ������������������Ϊ11��������������
        2. ����ĵ�һ��Ϊѧ��
    */
    int** getMatA(int row);

    /*
        ����B������
        1. ������������������Ϊ11����Ӧ����A��������������
    */
    int** getMatB(int col);

    /*
        ϡ�����A������
        1. ������������������Ϊ11��
        2. ����ϡ���
    */
    int** getSparseMatA(int row_size, double sparsity);

    /*
        ϡ�����B������
        1. ������������������Ϊ11��
        2. ����ϡ���
    */
    int** getSparseMatB(int col_size, double sparsity);

    /*
        �Ծ������Ԥ����
        1. ����[x0-average(x1,��,x8)]/ average(x1,��,x8)  <  10%�������򲻸ı�M(x0, y0)��ֵ����֮���þ�ֵ����
    */
    void meanFilter(int** mat, int row_size, int col_size);

    /*
        ͨ�þ���˷�
        1. �������A
        2. �������B
    */
    int** product(int** matA, int rowNumA, int** matB, int colNumB);

    /*
        ϡ�����˷�
        1. �������A
        2. �������B
    */
    int** sparseProduct(int** matA, int rowNumA, int** matB, int colNumB);

};
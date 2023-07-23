#pragma once
#include "sparseMatbody.h"
#include <iostream>
#include <mpi.h>
using namespace std;

class MySparseMat {
private:
    /*
        获取稀疏矩阵
        1. 输入行数
        2. 输入列数
        3. 输入稀疏度
    */
    int** getSparseMatrix(int row_size, int col_size, double sparsity);

    // 封装MPI数据类型的创建
    MPI_Datatype createMPI_Triple() {
        MPI_Datatype MPI_TRIPLE;
        int blocklengths[3] = { 1, 1, 1 };
        MPI_Aint displacements[3];
        MPI_Datatype types[3] = { MPI_INT, MPI_INT, MPI_INT };
        MPI_Aint baseaddr, addr1, addr2;
        MPI_Get_address(&((Triple*)0)->i, &baseaddr);
        MPI_Get_address(&((Triple*)0)->j, &addr1);
        MPI_Get_address(&((Triple*)0)->e, &addr2);
        displacements[0] = 0;
        displacements[1] = addr1 - baseaddr;
        displacements[2] = addr2 - baseaddr;
        MPI_Type_create_struct(3, blocklengths, displacements, types, &MPI_TRIPLE);
        MPI_Type_commit(&MPI_TRIPLE);
        return MPI_TRIPLE;
    }

public:
    /*
        打印矩阵
    */
    void display(RLSMatrix M);

    /*
        把二维矩阵压缩行逻辑链接顺序表形式
    */
    RLSMatrix compressSparseMat(int** mat, int row, int col);

    /*
        串行计算稀疏矩阵的点积
    */
    RLSMatrix product(RLSMatrix A, RLSMatrix B, RLSMatrix C);

    /*
        并行计算稀疏矩阵的点积
    */
    RLSMatrix parallelProduct(RLSMatrix A, RLSMatrix B, RLSMatrix C);
};
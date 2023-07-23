#pragma once
#include "sparseMatbody.h"
#include <iostream>
#include <mpi.h>
using namespace std;

class MySparseMat {
private:
    /*
        ��ȡϡ�����
        1. ��������
        2. ��������
        3. ����ϡ���
    */
    int** getSparseMatrix(int row_size, int col_size, double sparsity);

    // ��װMPI�������͵Ĵ���
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
        ��ӡ����
    */
    void display(RLSMatrix M);

    /*
        �Ѷ�ά����ѹ�����߼�����˳�����ʽ
    */
    RLSMatrix compressSparseMat(int** mat, int row, int col);

    /*
        ���м���ϡ�����ĵ��
    */
    RLSMatrix product(RLSMatrix A, RLSMatrix B, RLSMatrix C);

    /*
        ���м���ϡ�����ĵ��
    */
    RLSMatrix parallelProduct(RLSMatrix A, RLSMatrix B, RLSMatrix C);
};
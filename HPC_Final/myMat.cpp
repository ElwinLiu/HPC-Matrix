#include "myMat.h"

void MyMat::meanFilter(int** mat, int row_size, int col_size)
{   
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 串行则跳出
    if (size < 2) return;

    int rows_per_process = row_size / (size - 1);
    int extra_rows = row_size % (size - 1);

    if (rank == 0) { // 主进程分发数据
        for (int dest_rank = 1; dest_rank < size; dest_rank++) {
            int dest_start_row = (dest_rank - 1) * rows_per_process;
            int dest_end_row = dest_rank * rows_per_process;

            if (dest_rank == size - 1) {
                dest_end_row += extra_rows;
            }

            // 把矩阵数据发送给目标进程
            if (dest_start_row > 0) { // 多发上面一行
                MPI_Send(&mat[dest_start_row - 1][0], col_size, MPI_INT, dest_rank, 0, MPI_COMM_WORLD);
            }
            for (int i = dest_start_row; i < dest_end_row; i++) {
                MPI_Send(&mat[i][0], col_size, MPI_INT, dest_rank, 0, MPI_COMM_WORLD);
            }
            if (dest_rank < size - 1) { // 多发下面一行
                MPI_Send(&mat[dest_end_row][0], col_size, MPI_INT, dest_rank, 0, MPI_COMM_WORLD);
            }
        }
    }
    else { // 子进程获取并处理数据
        int start_row = (rank - 1) * rows_per_process;
        int end_row = rank * rows_per_process;
        if (rank == size - 1) {
            end_row += extra_rows;
        }
        int sub_row_size = rows_per_process;
        if (rank == size - 1) {
            sub_row_size = end_row - start_row;
        }
        if (start_row > 0) sub_row_size++;
        if (end_row < row_size) sub_row_size++;

        mat = new int* [sub_row_size];
        int** resultMat = new int* [sub_row_size];

        for (int i = 0; i < sub_row_size; i++) {
            mat[i] = new int[col_size];
            resultMat[i] = new int[col_size];
            MPI_Recv(&mat[i][0], col_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < col_size; j++) {
                resultMat[i][j] = mat[i][j];
            }
        }

        int start_line = 1;
        if (start_row == 0) start_line = 0;
        int end_line = sub_row_size - 2;
        if (end_row == row_size) end_line = sub_row_size - 1;

        //if (rank == 2) {
        //    cout << "\n start_row = " << start_row << endl;
        //    cout << "\n end_row = " << end_row << endl;
        //    cout << "\n start_line = " << start_line << endl;
        //    cout << "\n end_line = " << end_line << endl;
        //    cout << "\n sub_row_size = " << sub_row_size << endl;
        //}

        for (int i = start_line; i <= end_line; i++) {
            for (int j = 0; j < col_size; j++) {
                int count = 0;
                int sum = 0;
                for (int m = i - 1; m <= i + 1; m++) {
                    for (int n = j - 1; n <= j + 1; n++) {
                        if (m >= 0 && m < sub_row_size && n >= 0 && n < col_size && !(m == i && n == j)) {
                            // if (rank == 1 && i == 0 && j == 0) cout << "!!! " << mat[m][n];
                            sum += mat[m][n];
                            count++;
                        }
                    }
                }
                float average = static_cast<float>(sum) / count;
                int centerElement = mat[i][j];
                // if (rank == 1 && i == 0 && j == 0) cout << "count: " << count << " avg: " << average << endl;
                if (abs(centerElement - average) / average >= 0.1) {
                    resultMat[i][j] = static_cast<int>(average);
                }
            }
        }

        for (int i = start_line; i <= end_line; i++) {
            MPI_Send(&resultMat[i][0], col_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }

        //if (rank == 2) {
        //    printMatrix(resultMat, sub_row_size, col_size);
        //    cout << endl;
        //}

        // Free memory
        for (int i = 0; i < sub_row_size; i++) {
            delete[] mat[i];
        }
        delete[] mat;

        for (int i = 0; i < sub_row_size; i++) {
            delete[] resultMat[i];
        }
        delete[] resultMat;
    }

    if (rank == 0) {
        int** resultMatrix = new int* [row_size];
        for (int i = 0; i < row_size; i++) {
            resultMatrix[i] = new int[col_size];
            for (int j = 0; j < col_size; j++) {
                resultMatrix[i][j] = mat[i][j];
            }
        }

        for (int dest_rank = 1; dest_rank < size; dest_rank++) {
            int dest_start_row = (dest_rank - 1) * rows_per_process;
            int dest_end_row = dest_rank * rows_per_process;
            if (dest_rank == size - 1) {
                dest_end_row += extra_rows;
            }

            int subrow_size = dest_end_row - dest_start_row;
            int** subresultMatrix = new int* [subrow_size];
            /*cout << "receving process：" << dest_rank << "//" << subrow_size << endl;*/
            for (int i = 0; i < subrow_size; i++) {
                subresultMatrix[i] = new int[col_size];
                MPI_Recv(&subresultMatrix[i][0], col_size, MPI_INT, dest_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                /*cout << "debug：" << dest_rank << "//" << i << endl;*/
                /*if (rank == 1) for (int k = 0; k < col_size; k++) cout << "** " << subresultMatrix[i][k] << " " << endl;*/
            }

            // Copy the subresultMatrix to the main resultMatrix
            for (int i = dest_start_row; i < dest_end_row; i++) {
                for (int j = 0; j < col_size; j++) {
                    /*cout << "receving process：" << i << "//" << j << endl;*/
                    resultMatrix[i][j] = subresultMatrix[i - dest_start_row][j];
                }
            }

            // Free memory
            for (int i = 0; i < subrow_size; i++) {
                delete[] subresultMatrix[i];
            }
            delete[] subresultMatrix;
        }

        // printMatrix(resultMatrix, row_size, col_size);

        // Copy the resultMatrix to the original mat array
        for (int i = 0; i < row_size; i++) {
            for (int j = 0; j < col_size; j++) {
                mat[i][j] = resultMatrix[i][j];
            }
        }

        // Free memory
        for (int i = 0; i < row_size; i++) {
            delete[] resultMatrix[i];
        }
        delete[] resultMatrix;
    }
}

int** MyMat::product(int** matA, int rowNumA, int** matB, int colNumB)
{
    int rank, size;
    int common_size = 11;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = rowNumA / size;

    int* A = nullptr, *B = new int[common_size * colNumB], * C = nullptr;
    int* subA = new int[rows_per_process * common_size];
    int* subC = new int[rows_per_process * colNumB];

    // 1. 初始化一维矩阵
    if (rank == 0) {
        A = new int[rowNumA * common_size];
        int index = 0;
        for (int i = 0; i < rowNumA; i++) {
            for (int j = 0; j < common_size; j++) {
                A[index++] = matA[i][j];
            }
        }

        B = new int[common_size * colNumB];
        index = 0;
        for (int i = 0; i < common_size; i++) {
            for (int j = 0; j < colNumB; j++) {
                B[index++] = matB[i][j];
            }
        }
        C = new int[rowNumA * colNumB];
    }
    // 2. 阻塞所有进程，保证所有进程所需变量初始化成功
    MPI_Barrier(MPI_COMM_WORLD);

    // 3. 将矩阵A拆分发送给各个进程，Scatter()方法发送对象包括父进程本身
    MPI_Scatter(A, rows_per_process * common_size, MPI_INT, subA, rows_per_process * common_size, MPI_INT, 0, MPI_COMM_WORLD);
    // 4. 将矩阵B完整地广播出去
    MPI_Bcast(B, common_size * colNumB, MPI_INT, 0, MPI_COMM_WORLD);

    //cout << "this is process:" << rank << endl;

    // 5. 单个进程对矩阵的处理
    matMultiplyWithSingleThread(subA, B, subC, rows_per_process, common_size, colNumB);

    // 6. 汇集各个进程的结果
    MPI_Gather(subC, rows_per_process * colNumB, MPI_INT, C, rows_per_process * colNumB, MPI_INT, 0, MPI_COMM_WORLD);

    // 7. 用进程0处理剩余的行
    int remainRowsStartId = rows_per_process * size;
    if (rank == 0 && remainRowsStartId < rowNumA) {
        int remainRows = rowNumA - remainRowsStartId;
        matMultiplyWithSingleThread(A + remainRowsStartId * common_size, B, C + remainRowsStartId * colNumB, remainRows, common_size, colNumB);
    }

    delete[] subA;
    delete[] subC;
    delete[] B;

    int** matC = nullptr;
    if (rank == 0) {
        // 主进程获取结果矩阵
        matC = new int* [rowNumA];
        for (int i = 0; i < rowNumA; i++) {
            matC[i] = new int[colNumB];
            for (int j = 0; j < colNumB; j++) {
                matC[i][j] = C[i * colNumB + j];
            }
        }
        delete[] A;
        delete[] C;
    }

    return matC;
}

int** MyMat::sparseProduct(int** matA, int rowNumA, int** matB, int colNumB)
{
    return nullptr;
}

int** MyMat::getMat(int row, int col)
{
    // 初始化随机数生成器
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dis(0, 100);

    int** inputMatrix = new int* [row];
    for (int i = 0; i < row; ++i) {
        inputMatrix[i] = new int[col];
        for (int j = 0; j < col; ++j) {
            // 随机生成数值
            inputMatrix[i][j] = dis(gen);
        }
    }

    return inputMatrix;
}

int** MyMat::getMatA(int row)
{
    int** ret = getMat(row, 11);
    int digits[] = { 2, 0, 2, 0, 1, 0, 0, 3, 7, 2, 9 };
    // 把第一行设置为学号
    for (int i = 0; i < 11; i++) {
        ret[0][i] = digits[i];
    }
    return ret;
}

int** MyMat::getMatB(int col)
{
    return getMat(11, col);
}

void MyMat::printMatrix(int** matrix, int row_size, int col_size) {
    for (int i = 0; i < row_size; ++i) {
        cout << "第" << i << "行: ";
        for (int j = 0; j < col_size; ++j) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

int** MyMat::getSparseMatrix(int row_size, int col_size, double sparsity)
{
    int** sparseMatrix = new int* [row_size];

    for (int row = 0; row < row_size; row++) {
        sparseMatrix[row] = new int[col_size];

        for (int col = 0; col < col_size; col++) {
            // 生成随机值，根据给定的稀疏度确定是否为零
            if (static_cast<double>(rand()) / RAND_MAX > sparsity) {
                sparseMatrix[row][col] = 0;
            }
            else {
                // 生成非零随机值，范围在1到9之间
                sparseMatrix[row][col] = rand() % 100 + 1;
            }
        }
    }

    return sparseMatrix;
}

void MyMat::matMultiplyWithSingleThread(int* A, int* B, int* matResult, int m, int p, int n)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float temp = 0;
            for (int k = 0; k < p; k++) {
                temp += A[i * p + k] * B[k * n + j];
            }
            matResult[i * n + j] = temp;
        }
    }
}

int** MyMat::getSparseMatA(int row_size, double sparsity)
{
    int** ret = getSparseMatrix(row_size, 11, sparsity);
    int digits[] = { 2, 0, 2, 0, 1, 0, 0, 3, 7, 2, 9 };
    // 把第一行设置为学号
    for (int i = 0; i < 11; i++) {
        ret[0][i] = digits[i];
    }
    return ret;
}

int** MyMat::getSparseMatB(int col_size, double sparsity)
{
    return getSparseMatrix(11, col_size, sparsity);
}
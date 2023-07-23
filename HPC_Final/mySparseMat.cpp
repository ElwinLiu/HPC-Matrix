#include "mySparseMat.h"

void MySparseMat::display(RLSMatrix M)
{
	cout << "test:::::" << M.mu << " && " << M.nu << endl;
	int i, j, k;
	for (i = 1; i <= M.mu; i++) {
		for (j = 1; j <= M.nu; j++) {
			int value = 0;
			//输出前 mu - 1 行矩阵
			if (i + 1 <= M.mu) {
				for (k = M.rowOffset[i]; k < M.rowOffset[i + 1]; k++) {
					if (i == M.data[k].i && j == M.data[k].j) {
						printf("%d ", M.data[k].e);
						value = 1;
						break;
					}
				}
				if (value == 0) {
					printf("0 ");
				}
			}
			//输出矩阵最后一行的数据
			else {
				for (k = M.rowOffset[i]; k <= M.tu; k++) {
					if (i == M.data[k].i && j == M.data[k].j) {
						printf("%d ", M.data[k].e);
						value = 1;
						break;
					}
				}
				if (value == 0) {
					printf("0 ");
				}
			}
		}
		printf("\n");
	}
}

RLSMatrix MySparseMat::compressSparseMat(int** mat, int row, int col)
{
	RLSMatrix comMat;
	comMat.mu = row;
	comMat.nu = col;
	Triple data[1000]; // 从1开始存储
	int rowOffset[1000] = { 0 }; // 从1开始存储
	int tu = 0; // 表示非零元素的个数


	for (int i = 0; i < row; i++) {
		bool flag = true; // 如果找到了该行第一个元素，则设置为false
		for (int j = 0; j < col; j++) {
			if (mat[i][j] != 0) { // 记录非零元素
				tu++;
				data[tu].e = mat[i][j];
				data[tu].i = i + 1;
				data[tu].j = j + 1;

				if (flag) { // 记录第一个非零元素在data数组里的位置
					flag = false;
					rowOffset[i + 1] = tu;
				}
			}
		}
		if (flag == true) { //遍历整行没有找到非零元素，则将offset值设置为tu+1
			rowOffset[i + 1] = -1;
		}
	}

	for (int i = 0; i < 1000; i++) {
		if (rowOffset[i] == -1) {
			rowOffset[i] = tu + 1;
		}
	}

	comMat.mu = row;
	comMat.nu = col;
	comMat.tu = tu;

	// 将data数组复制到comMat的data数组中
	for (int i = 1; i <= tu; i++) {
		comMat.data[i] = data[i];
	}

	// 将rowOffset数组复制到comMat的rowOffset数组中
	for (int i = 1; i <= row; i++) {
		comMat.rowOffset[i] = rowOffset[i];
	}

	return comMat;
}

RLSMatrix MySparseMat::product(RLSMatrix A, RLSMatrix B, RLSMatrix C)
{
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank != 0) {
		return RLSMatrix();
	}

	// 初始化矩阵C的各个参数
	C.mu = A.mu;
	C.nu = B.nu;
	C.tu = 0;

	/*
		判断矩阵点积的合法性
		要求1. 矩阵A的列数=矩阵B的行数
		要求2. 两个矩阵必须均存在非零值，否则乘法运算没有意义
	*/
	if (A.nu != B.mu || A.tu * B.tu == 0) {
		return C;
	}

	int arow, ccol;

	//遍历矩阵A的每一行
	for (arow = 1; arow <= A.mu; arow++)
	{
		//创建一个临时存储乘积结果的数组，且初始化为0，遍历每次都需要清空
		int ctemp[1000] = {};
		C.rowOffset[arow] = C.tu + 1;
		//根据行数，在三元组表中找到该行所有的非0元素的位置
		int tp;
		if (arow < A.mu)
			tp = A.rowOffset[arow + 1];//获取矩阵A的下一行第一个非零元素在data数组中位置
		else
			tp = A.tu + 1;//若当前行是最后一行，则取最后一个元素+1

		int p;
		int brow;
		//遍历当前行的所有的非0元素
		for (p = A.rowOffset[arow]; p < tp; p++)
		{
			brow = A.data[p].j;//取该非0元素的列数，便于去B中找对应的做乘积的非0元素
			int t;
			// 判断如果对于A中非0元素，找到矩阵B中做乘法的那一行中的所有的非0元素
			if (brow < B.mu)
				t = B.rowOffset[brow + 1];
			else
				t = B.tu + 1;
			int q;
			//遍历找到的对应的非0元素，开始做乘积运算
			for (q = B.rowOffset[brow]; q < t; q++)
			{
				//得到的乘积结果，每次和ctemp数组中相应位置的数值做加和运算
				ccol = B.data[q].j;
				ctemp[ccol] += A.data[p].e * B.data[q].e;
			}
		}
		//矩阵C的行数等于矩阵A的行数，列数等于矩阵B的列数，所以，得到的ctemp存储的结果，也会在C的列数的范围内
		for (ccol = 1; ccol <= C.nu; ccol++)
		{
			//由于结果可以是0，而0不需要存储，所以在这里需要判断
			if (ctemp[ccol])
			{
				//不为0，则记录矩阵中非0元素的个数的变量tu要+1；且该值不能超过存放三元素数组的空间大小
				if (++C.tu > 1000)
					return C;
				else {
					C.data[C.tu].e = ctemp[ccol];
					C.data[C.tu].i = arow;
					C.data[C.tu].j = ccol;
				}
			}
		}
	}
	return C;
}

RLSMatrix MySparseMat::parallelProduct(RLSMatrix A, RLSMatrix B, RLSMatrix C)
{
	int rank, size;
	int rowA, colA, tA, rowB, colB, tB;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// 最终回收的数据
	Triple* dataC = nullptr;
	int* rowOffsetC = nullptr;
	int rowC(0), colC(0), tC(0);
	// 子进程进行处理的数据
	Triple* sub_dataC = nullptr;
	int* sub_rowOffsetC = nullptr;
	int sub_tC(0);

	if (rank == 0) {
		rowA = A.mu;
		colA = A.nu;
		tA = A.tu;
		rowB = B.mu;
		colB = B.nu;
		tB = B.tu;
	}

	// 将矩阵公用部分的参数广播出去
	MPI_Bcast(&rowA, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&colA, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&rowB, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&colB, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// 创建MPI的三元组数据类型
	MPI_Datatype MPI_TRIPLE = createMPI_Triple();

	int rows_per_process = A.mu / (size - 1);
	int extra_rows = A.mu % (size - 1);

	/*
		主进程分发数据
		1. data数据
		2. rowOffset数据
		3. 其他数据在子进程中重新计算获取（不占用主进程的计算资源）
	*/
	if (rank == 0) {
		for (int dest_rank = 1; dest_rank < size; dest_rank++) {
			// 计算起止行数（从1开始）
			int start_row = (dest_rank - 1) * rows_per_process + 1;
			int end_row = dest_rank * rows_per_process;
			if (dest_rank == size - 1) { // 最后一个进程包揽多余的行
				end_row += extra_rows;
			}

			// 计算对应的非零元素在表中的范围
			int start_data = A.rowOffset[start_row];
			int len_data = A.rowOffset[end_row] - start_data;

			// 分发矩阵A的数据
			MPI_Send(&A.rowOffset[start_row], end_row - start_row + 1, MPI_INT, dest_rank, 0, MPI_COMM_WORLD);
			MPI_Send(&A.data[start_data], len_data, MPI_TRIPLE, dest_rank, 0, MPI_COMM_WORLD);
			

			// 分发矩阵B的数据
			MPI_Send(&B.rowOffset[1], rowB, MPI_INT, dest_rank, 0, MPI_COMM_WORLD);
			MPI_Send(&B.data[1], tB, MPI_TRIPLE, dest_rank, 0, MPI_COMM_WORLD);
			
		}
	}
	else {
		int rows = rows_per_process;
		// 最后一个进程包揽多余的行
		if (rank == size - 1) {
			rows += extra_rows;
		}
		// 计算起止行数
		int start_row = (rank - 1) * rows_per_process + 1;
		int end_row = rank * rows_per_process;
		int len_data; // data数组中有效的非零元素个数

		// 按照行数分配Offset
		int* rowOffset = new int[end_row - start_row + 1];
		// 按比例分配data的长度
		Triple* data = new Triple[1000 * rows / rowA];

		MPI_Recv(&rowOffset[1], end_row - start_row + 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		len_data = rowOffset[end_row - start_row + 1] - rowOffset[1];
		MPI_Recv(&data[1], len_data, MPI_TRIPLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		// 处理rowOffset数组
		int delta = start_row - 1; // 另起始行数为第一行
		for (int i = 0; i < end_row - start_row + 1; i++) {
			rowOffset[i + 1] -= delta;
		}

		// 定义分配到的子矩阵A
		RLSMatrix subA;
		for (int i = 1; i <= end_row - start_row + 1; i++) {
			subA.data[i] = data[i];
		}
		for (int i = 1; i <= len_data; i++) {
			subA.rowOffset[i] = rowOffset[i];
		}
		subA.mu = rows;
		subA.nu = colA;
		subA.tu = len_data;

		// 定义分到到的完整矩阵B
		RLSMatrix B;
		MPI_Recv(&B.rowOffset[1], rowB, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&B.data[1], tB, MPI_TRIPLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		B.mu = rowB;
		B.nu = colB;
		B.tu = tB;

		// 获取子矩阵
		RLSMatrix res;
		res = product(subA, B, res);

		dataC = new Triple[res.tu + 1];
		for (int i = 1; i <= res.tu; i++) {
			sub_dataC[i] = res.data[i];
		}

		rowOffsetC = new int[res.mu + 1];
		for (int i = 1; i <= res.mu; i++) {
			sub_rowOffsetC[i] = res.rowOffset[i];
		}

		sub_tC = res.tu;
	}

	// 主进程收集各子进程的计算结果
	if (rank == 0) {
		MPI_Gather(dataC, tC, MPI_TRIPLE, sub_dataC, tC, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Gather(rowOffsetC, rowA, MPI_TRIPLE, sub_rowOffsetC, rowA, MPI_INT, 0, MPI_COMM_WORLD);
		rowC = rowA;
		colC = colB;
		RLSMatrix resC;
		for (int i = 0; i < tC; i++) {
			resC.data[i] = dataC[i];
		}

		for (int i = 0; i < rowA; i++) {
			resC.rowOffset[i] = rowOffsetC[i];
		}

		resC.mu = rowC;
		resC.nu = colC;
		resC.tu = tC;

		return resC;
	}
	return RLSMatrix();
}

int** MySparseMat::getSparseMatrix(int row_size, int col_size, double sparsity)
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
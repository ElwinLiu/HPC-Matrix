#include "mySparseMat.h"

void MySparseMat::display(RLSMatrix M)
{
	cout << "test:::::" << M.mu << " && " << M.nu << endl;
	int i, j, k;
	for (i = 1; i <= M.mu; i++) {
		for (j = 1; j <= M.nu; j++) {
			int value = 0;
			//���ǰ mu - 1 �о���
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
			//����������һ�е�����
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
	Triple data[1000]; // ��1��ʼ�洢
	int rowOffset[1000] = { 0 }; // ��1��ʼ�洢
	int tu = 0; // ��ʾ����Ԫ�صĸ���


	for (int i = 0; i < row; i++) {
		bool flag = true; // ����ҵ��˸��е�һ��Ԫ�أ�������Ϊfalse
		for (int j = 0; j < col; j++) {
			if (mat[i][j] != 0) { // ��¼����Ԫ��
				tu++;
				data[tu].e = mat[i][j];
				data[tu].i = i + 1;
				data[tu].j = j + 1;

				if (flag) { // ��¼��һ������Ԫ����data�������λ��
					flag = false;
					rowOffset[i + 1] = tu;
				}
			}
		}
		if (flag == true) { //��������û���ҵ�����Ԫ�أ���offsetֵ����Ϊtu+1
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

	// ��data���鸴�Ƶ�comMat��data������
	for (int i = 1; i <= tu; i++) {
		comMat.data[i] = data[i];
	}

	// ��rowOffset���鸴�Ƶ�comMat��rowOffset������
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

	// ��ʼ������C�ĸ�������
	C.mu = A.mu;
	C.nu = B.nu;
	C.tu = 0;

	/*
		�жϾ������ĺϷ���
		Ҫ��1. ����A������=����B������
		Ҫ��2. ���������������ڷ���ֵ������˷�����û������
	*/
	if (A.nu != B.mu || A.tu * B.tu == 0) {
		return C;
	}

	int arow, ccol;

	//��������A��ÿһ��
	for (arow = 1; arow <= A.mu; arow++)
	{
		//����һ����ʱ�洢�˻���������飬�ҳ�ʼ��Ϊ0������ÿ�ζ���Ҫ���
		int ctemp[1000] = {};
		C.rowOffset[arow] = C.tu + 1;
		//��������������Ԫ������ҵ��������еķ�0Ԫ�ص�λ��
		int tp;
		if (arow < A.mu)
			tp = A.rowOffset[arow + 1];//��ȡ����A����һ�е�һ������Ԫ����data������λ��
		else
			tp = A.tu + 1;//����ǰ�������һ�У���ȡ���һ��Ԫ��+1

		int p;
		int brow;
		//������ǰ�е����еķ�0Ԫ��
		for (p = A.rowOffset[arow]; p < tp; p++)
		{
			brow = A.data[p].j;//ȡ�÷�0Ԫ�ص�����������ȥB���Ҷ�Ӧ�����˻��ķ�0Ԫ��
			int t;
			// �ж��������A�з�0Ԫ�أ��ҵ�����B�����˷�����һ���е����еķ�0Ԫ��
			if (brow < B.mu)
				t = B.rowOffset[brow + 1];
			else
				t = B.tu + 1;
			int q;
			//�����ҵ��Ķ�Ӧ�ķ�0Ԫ�أ���ʼ���˻�����
			for (q = B.rowOffset[brow]; q < t; q++)
			{
				//�õ��ĳ˻������ÿ�κ�ctemp��������Ӧλ�õ���ֵ���Ӻ�����
				ccol = B.data[q].j;
				ctemp[ccol] += A.data[p].e * B.data[q].e;
			}
		}
		//����C���������ھ���A���������������ھ���B�����������ԣ��õ���ctemp�洢�Ľ����Ҳ����C�������ķ�Χ��
		for (ccol = 1; ccol <= C.nu; ccol++)
		{
			//���ڽ��������0����0����Ҫ�洢��������������Ҫ�ж�
			if (ctemp[ccol])
			{
				//��Ϊ0�����¼�����з�0Ԫ�صĸ����ı���tuҪ+1���Ҹ�ֵ���ܳ��������Ԫ������Ŀռ��С
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

	// ���ջ��յ�����
	Triple* dataC = nullptr;
	int* rowOffsetC = nullptr;
	int rowC(0), colC(0), tC(0);
	// �ӽ��̽��д��������
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

	// �������ò��ֵĲ����㲥��ȥ
	MPI_Bcast(&rowA, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&colA, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&rowB, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&colB, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// ����MPI����Ԫ����������
	MPI_Datatype MPI_TRIPLE = createMPI_Triple();

	int rows_per_process = A.mu / (size - 1);
	int extra_rows = A.mu % (size - 1);

	/*
		�����̷ַ�����
		1. data����
		2. rowOffset����
		3. �����������ӽ��������¼����ȡ����ռ�������̵ļ�����Դ��
	*/
	if (rank == 0) {
		for (int dest_rank = 1; dest_rank < size; dest_rank++) {
			// ������ֹ��������1��ʼ��
			int start_row = (dest_rank - 1) * rows_per_process + 1;
			int end_row = dest_rank * rows_per_process;
			if (dest_rank == size - 1) { // ���һ�����̰����������
				end_row += extra_rows;
			}

			// �����Ӧ�ķ���Ԫ���ڱ��еķ�Χ
			int start_data = A.rowOffset[start_row];
			int len_data = A.rowOffset[end_row] - start_data;

			// �ַ�����A������
			MPI_Send(&A.rowOffset[start_row], end_row - start_row + 1, MPI_INT, dest_rank, 0, MPI_COMM_WORLD);
			MPI_Send(&A.data[start_data], len_data, MPI_TRIPLE, dest_rank, 0, MPI_COMM_WORLD);
			

			// �ַ�����B������
			MPI_Send(&B.rowOffset[1], rowB, MPI_INT, dest_rank, 0, MPI_COMM_WORLD);
			MPI_Send(&B.data[1], tB, MPI_TRIPLE, dest_rank, 0, MPI_COMM_WORLD);
			
		}
	}
	else {
		int rows = rows_per_process;
		// ���һ�����̰����������
		if (rank == size - 1) {
			rows += extra_rows;
		}
		// ������ֹ����
		int start_row = (rank - 1) * rows_per_process + 1;
		int end_row = rank * rows_per_process;
		int len_data; // data��������Ч�ķ���Ԫ�ظ���

		// ������������Offset
		int* rowOffset = new int[end_row - start_row + 1];
		// ����������data�ĳ���
		Triple* data = new Triple[1000 * rows / rowA];

		MPI_Recv(&rowOffset[1], end_row - start_row + 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		len_data = rowOffset[end_row - start_row + 1] - rowOffset[1];
		MPI_Recv(&data[1], len_data, MPI_TRIPLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		// ����rowOffset����
		int delta = start_row - 1; // ����ʼ����Ϊ��һ��
		for (int i = 0; i < end_row - start_row + 1; i++) {
			rowOffset[i + 1] -= delta;
		}

		// ������䵽���Ӿ���A
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

		// ����ֵ�������������B
		RLSMatrix B;
		MPI_Recv(&B.rowOffset[1], rowB, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&B.data[1], tB, MPI_TRIPLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		B.mu = rowB;
		B.nu = colB;
		B.tu = tB;

		// ��ȡ�Ӿ���
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

	// �������ռ����ӽ��̵ļ�����
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
			// �������ֵ�����ݸ�����ϡ���ȷ���Ƿ�Ϊ��
			if (static_cast<double>(rand()) / RAND_MAX > sparsity) {
				sparseMatrix[row][col] = 0;
			}
			else {
				// ���ɷ������ֵ����Χ��1��9֮��
				sparseMatrix[row][col] = rand() % 100 + 1;
			}
		}
	}

	return sparseMatrix;
}
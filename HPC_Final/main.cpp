#include "myMat.h"
#include "mySparseMat.h"
#include <chrono>
#include <fstream>

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int rowNumA;
	int colNumB;
	MyMat myMat;
	int** matA = nullptr;
	int** matB = nullptr;

	if (rank == 0) {
		cout <<
			"+---------------------------------------------------------------------------------+\n"
			"|                                                                                 |\n"
			"|                                                                                 |\n"
			"|  ˵�����ó���������111203�����꣬���������̿���ʦ�Ŀγ������Ŀ    ��������   |\n"
			"|  ����1 ʹ����3*3��ֵ�˲��Ծ������Ԥ���� ��2 ���ͨ�þ���˷� ��3 ���ϡ�����  |\n"
			"|  �˷���                                                                         |\n"
			"|                                                                                 |\n"
			"|  ע�⣺�����Ϊ A��B���� ����A����ĵ�һ�б���Ϊ���˵�ѧ��20201003729�����     |\n"
			"|  �����涨�˾��� A������Ϊ 11������ B������Ϊ 11.�����������û�ʹ�øó���ʱֻ    |\n"
			"|  ��Ҫ������� A�������;���  B��������                                          |\n"
			"|                                                                                 |\n"
			"|                                                                                 |\n"
			"|                                                                                 |\n"
			"|                   +------------------------------+                              |\n"
			"|                   |            ����              |                              |\n"
			"|                   |                              |                              |\n"
			"|                   |  ����A���� ��---------       |                              |\n"
			"|                   |                              |                              |\n"
			"|                   |  ����B���� ��---------       |                              |\n"
			"|                   |                              |                              |\n"
			"|                   +------------------------------+                              |\n"
			"|                                                                                 |\n"
			"|                                                                                 |\n"
			"+---------------------------------------------------------------------------------+\n"
			<< endl;
		cout << "����A������";
		cin >> rowNumA;
		cout << endl;
		cout << "����B������";
		cin >> colNumB;
		cout << endl;

		// ���ɾ���A
		matA = myMat.getMatA(rowNumA);
		cout << "matrix A:" << endl;
		// myMat.printMatrix(matA, rowNumA, 11);
		cout << endl;

		// ���ɾ���B
		matB = myMat.getMatB(colNumB);
		cout << "matrix B:" << endl;
		// myMat.printMatrix(matB, 11, colNumB);
		cout << endl;



	}

	// �ѻ�ȡ�ľ����С��Ϣ�㲥�������߳�
	MPI_Bcast(&rowNumA, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&colNumB, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int input = 0;
	while (input != 7) {
		if (rank == 0) {
			cout <<
				"+--------------------------------------------------------+\n"
				"|                                                        |\n"
				"|    ��������                                            |\n"
				"|                                                        |\n"
				"|    1. ��������ϡ�����A��                              |\n"
				"|    2. ��������ϡ�����B��                              |\n"
				"|    3. �Ծ���A����Ԥ����                              |\n"
				"|    4. �Ծ���B����Ԥ����                              |\n"
				"|    5. ͨ�þ���˷�A * B��                              |\n"
				"|    6. ϡ�����˷�A * B��                              |\n"
				"|    7. �˳���                                           |\n"
				"|                                                        |\n"
				"+--------------------------------------------------------+\n"
				<< endl;
			cin >> input;
		}

		MPI_Bcast(&input, 1, MPI_INT, 0, MPI_COMM_WORLD);
		
		switch (input) {
		case 1: {
			if (rank == 0) { // ֻ�ø���������ϡ�����
				for (int i = 0; i < rowNumA; i++) {
					delete[] matA[i];
				}
				delete[] matA;
				matA = myMat.getSparseMatA(rowNumA, 0.15);
				myMat.printMatrix(matA, rowNumA, 11);
				cout << endl;
			}
			break;
		}
		case 2: { 
			if (rank == 0) { // ֻ�ø���������ϡ�����
				for (int i = 0; i < 11; i++) {
					delete[] matB[i];
				}
				delete[] matB;
				matB = myMat.getSparseMatB(colNumB, 0.15);
				myMat.printMatrix(matB, 11, colNumB);
				cout << endl;
			}
			break;
		}
		case 3: {
			// ��¼����ʼʱ��
			auto start_time = std::chrono::high_resolution_clock::now();
			myMat.meanFilter(matA, rowNumA, 11);
			MPI_Barrier(MPI_COMM_WORLD);
			if (rank == 0) {
				// ��¼�������ʱ��
				auto end_time = std::chrono::high_resolution_clock::now();
				// �����������ʱ��
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
				// �������ʱ�䵽��־�ļ�
				cout << duration << endl;
				//cout << "�Ծ���A����Ԥ�����Ժ�" << endl;
				//myMat.printMatrix(matA, rowNumA, 11);
				//cout << endl;
			}
			break;
		}
		case 4: {
			// ��¼����ʼʱ��
			auto start_time = std::chrono::high_resolution_clock::now();
			myMat.meanFilter(matB, 11, colNumB);
			MPI_Barrier(MPI_COMM_WORLD);
			if (rank == 0) {
				// ��¼�������ʱ��
				auto end_time = std::chrono::high_resolution_clock::now();
				// �����������ʱ��
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
				// �������ʱ�䵽��־�ļ�
				cout << duration << endl;
				//cout << "�Ծ���B����Ԥ�����Ժ�" << endl;
				//myMat.printMatrix(matB, 11, colNumB);
				//cout << endl;
			}
			break;
		}
		case 5: {
			// ��¼����ʼʱ��
			auto start_time = std::chrono::high_resolution_clock::now();
			int** C = myMat.product(matA, rowNumA, matB, colNumB);
			MPI_Barrier(MPI_COMM_WORLD);
			if (rank == 0) {
				// ��¼�������ʱ��
				auto end_time = std::chrono::high_resolution_clock::now();
				// �����������ʱ��
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
				cout << duration << endl;
				cout << "��ϡ�裩�����������" << endl;
				myMat.printMatrix(C, rowNumA, colNumB);
				cout << endl;
				for (int i = 0; i < rowNumA; i++) {
					delete[] C[i];
				}
				delete[] C;
			}
			break;
		}
		case 6: {
			MySparseMat sparseMat;
			// ʹ�����߼�˳���ѹ��ϡ�����
			RLSMatrix compressedMatA;
			RLSMatrix compressedMatB;
			RLSMatrix compressedMatC;
			// �����̶Ծ������ѹ������
			if (rank == 0) {
				RLSMatrix compressedMatA = sparseMat.compressSparseMat(matA, rowNumA, 11);
				RLSMatrix compressedMatB = sparseMat.compressSparseMat(matB, 11, colNumB);
				compressedMatC = sparseMat.product(compressedMatA, compressedMatB, compressedMatC);
			}
			
			// ��ӡ�˷����
			if (rank == 0) {
				sparseMat.display(compressedMatC);
			}
		}
		case 7: {break; }
		default: {break; }
		}
	}




	MPI_Finalize();
}
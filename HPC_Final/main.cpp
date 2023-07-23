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
			"|  说明：该程序作者是111203刘星雨，用于完成李程俊老师的课程设计题目    。内容有   |\n"
			"|  三：1 使用类3*3均值滤波对矩阵进行预处理 ，2 完成通用矩阵乘法 ，3 完成稀疏矩阵  |\n"
			"|  乘法。                                                                         |\n"
			"|                                                                                 |\n"
			"|  注意：矩阵分为 A、B两个 ，且A矩阵的第一行必须为本人的学号20201003729，这个     |\n"
			"|  条件规定了矩阵 A的列数为 11，矩阵 B的行数为 11.综上所述，用户使用该程序时只    |\n"
			"|  需要输入矩阵 A的行数和矩阵  B的列数！                                          |\n"
			"|                                                                                 |\n"
			"|                                                                                 |\n"
			"|                                                                                 |\n"
			"|                   +------------------------------+                              |\n"
			"|                   |            输入              |                              |\n"
			"|                   |                              |                              |\n"
			"|                   |  矩阵A行数 ：---------       |                              |\n"
			"|                   |                              |                              |\n"
			"|                   |  矩阵B行数 ：---------       |                              |\n"
			"|                   |                              |                              |\n"
			"|                   +------------------------------+                              |\n"
			"|                                                                                 |\n"
			"|                                                                                 |\n"
			"+---------------------------------------------------------------------------------+\n"
			<< endl;
		cout << "矩阵A行数：";
		cin >> rowNumA;
		cout << endl;
		cout << "矩阵B列数：";
		cin >> colNumB;
		cout << endl;

		// 生成矩阵A
		matA = myMat.getMatA(rowNumA);
		cout << "matrix A:" << endl;
		// myMat.printMatrix(matA, rowNumA, 11);
		cout << endl;

		// 生成矩阵B
		matB = myMat.getMatB(colNumB);
		cout << "matrix B:" << endl;
		// myMat.printMatrix(matB, 11, colNumB);
		cout << endl;



	}

	// 把获取的矩阵大小信息广播给所有线程
	MPI_Bcast(&rowNumA, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&colNumB, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int input = 0;
	while (input != 7) {
		if (rank == 0) {
			cout <<
				"+--------------------------------------------------------+\n"
				"|                                                        |\n"
				"|    输入数字                                            |\n"
				"|                                                        |\n"
				"|    1. 重新生成稀疏矩阵A。                              |\n"
				"|    2. 重新生成稀疏矩阵B。                              |\n"
				"|    3. 对矩阵A进行预处理。                              |\n"
				"|    4. 对矩阵B进行预处理。                              |\n"
				"|    5. 通用矩阵乘法A * B。                              |\n"
				"|    6. 稀疏矩阵乘法A * B。                              |\n"
				"|    7. 退出。                                           |\n"
				"|                                                        |\n"
				"+--------------------------------------------------------+\n"
				<< endl;
			cin >> input;
		}

		MPI_Bcast(&input, 1, MPI_INT, 0, MPI_COMM_WORLD);
		
		switch (input) {
		case 1: {
			if (rank == 0) { // 只用父进程生成稀疏矩阵
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
			if (rank == 0) { // 只用父进程生成稀疏矩阵
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
			// 记录程序开始时间
			auto start_time = std::chrono::high_resolution_clock::now();
			myMat.meanFilter(matA, rowNumA, 11);
			MPI_Barrier(MPI_COMM_WORLD);
			if (rank == 0) {
				// 记录程序结束时间
				auto end_time = std::chrono::high_resolution_clock::now();
				// 计算程序运行时间
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
				// 输出运行时间到日志文件
				cout << duration << endl;
				//cout << "对矩阵A进行预处理以后：" << endl;
				//myMat.printMatrix(matA, rowNumA, 11);
				//cout << endl;
			}
			break;
		}
		case 4: {
			// 记录程序开始时间
			auto start_time = std::chrono::high_resolution_clock::now();
			myMat.meanFilter(matB, 11, colNumB);
			MPI_Barrier(MPI_COMM_WORLD);
			if (rank == 0) {
				// 记录程序结束时间
				auto end_time = std::chrono::high_resolution_clock::now();
				// 计算程序运行时间
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
				// 输出运行时间到日志文件
				cout << duration << endl;
				//cout << "对矩阵B进行预处理以后：" << endl;
				//myMat.printMatrix(matB, 11, colNumB);
				//cout << endl;
			}
			break;
		}
		case 5: {
			// 记录程序开始时间
			auto start_time = std::chrono::high_resolution_clock::now();
			int** C = myMat.product(matA, rowNumA, matB, colNumB);
			MPI_Barrier(MPI_COMM_WORLD);
			if (rank == 0) {
				// 记录程序结束时间
				auto end_time = std::chrono::high_resolution_clock::now();
				// 计算程序运行时间
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
				cout << duration << endl;
				cout << "（稀疏）矩阵点积结果：" << endl;
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
			// 使用行逻辑顺序表压缩稀疏矩阵
			RLSMatrix compressedMatA;
			RLSMatrix compressedMatB;
			RLSMatrix compressedMatC;
			// 主进程对矩阵进行压缩处理
			if (rank == 0) {
				RLSMatrix compressedMatA = sparseMat.compressSparseMat(matA, rowNumA, 11);
				RLSMatrix compressedMatB = sparseMat.compressSparseMat(matB, 11, colNumB);
				compressedMatC = sparseMat.product(compressedMatA, compressedMatB, compressedMatC);
			}
			
			// 打印乘法结果
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
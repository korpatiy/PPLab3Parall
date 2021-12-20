#include <iostream>
#include <cmath>
#include <fstream>
#include "mpi.h"

using namespace std;

const int N = 191;
const double a = 1.0;
const double h = a / N;
const double tau = h * h / 2 / 2;
const double T = 1;
//Находим кол-во точек по шагам времени.
const int Nt = T / tau;
//лямбда
const double L = tau / h;
const double L2 = pow(L, 2);

//функция по варианту
double func(double x) {
    if (x <= 0.7 && x >= 0.5)
        return (9.0 * x - 4.5) / 0.2;
    if (x > 0.7 && x <= 0.9)
        return (-9.0 * x + 8.1) / 0.2;
    return 0.0;
}

double g(double x) { return 0.0; }

double countRightPrev(int currN, int currSize, const double *currU, double right_prev, int t) {
    return 2.0 * (1.0 - L2) * currU[t * currSize + currN]
           + L2 * (currU[t * currSize + currN - 1] + right_prev)
           - currU[(t - 1) * currSize + currN];
}

double countLeftPrev(int currSize, const double *currU, double left_prev, int t) {
    return 2.0 * (1.0 - L2) * currU[t * currSize]
           + L2 * (left_prev + currU[t * currSize + 1])
           - currU[(t - 1) * currSize];
}

//Умф по методичке
double umf(int currSize, const double *currU, int t, int i) {
    return 2.0 * (1.0 - L2) * currU[t * currSize + i] + L2 * (currU[t * currSize + i + 1] + currU[t * currSize + i - 1])
           - currU[(t - 1) * currSize + i];
}

int main(int argc, char **argv) {

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto time1 = MPI_Wtime();

    int dims[1] = {0};
    MPI_Dims_create(size, 1, dims);
    int periods[1] = {false};

    MPI_Comm comm_1D;
    MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, true, &comm_1D);

    //Координаты соседей
    int left;
    int right;
    //Нахождение соседей слева и справа
    MPI_Cart_shift(comm_1D, 0, 1, &left, &right);

    MPI_Comm_rank(comm_1D, &rank);

    //Нарезаем часть, которую будет обрабатывать каждый процессор
    auto matrixPart = (N + 1) / size;

    //Произвольный тип данных для строки
    MPI_Datatype MPI_RAW;
    MPI_Type_vector(1, matrixPart, matrixPart, MPI_DOUBLE, &MPI_RAW);
    MPI_Type_commit(&MPI_RAW);

    auto begIdx = matrixPart * rank;
    auto endIdx = (rank == size - 1 ? N : begIdx + matrixPart - 1);

    auto currN = endIdx - begIdx;
    auto currSize = currN + 1;
    auto *currU = new double[(Nt + 1) * currSize];

    //начальные и краевые условия
    int k = 0;
    int end = currN;
    if (begIdx == 0) {
        currU[currSize] = currU[0] = 0.0;
        k++;
    }

    if (endIdx == N) {
        currU[currSize + currN] = currU[currN] = 0.0;
        end--;
    }

    for (; k <= end; k++) {
        //x от 0 до 1 в k * h
        currU[k] = func((begIdx + k) * h); //y1
        currU[currSize + k] = currU[k] + tau * g(begIdx + k); //y2
    }

    MPI_Status status1, status2;

    double left_prev, right_prev;
    MPI_Send(&currU[currSize], 1, MPI_DOUBLE, left, 0, comm_1D);
    MPI_Send(&currU[currSize + currN], 1, MPI_DOUBLE, right, 0, comm_1D);
    MPI_Recv(&left_prev, 1, MPI_DOUBLE, left, 0, comm_1D, &status1);
    MPI_Recv(&right_prev, 1, MPI_DOUBLE, right, 0, comm_1D, &status2);

    //Идем по времени от 0 до T.
    for (int t = 1; t < Nt; t++) {
        auto currTIdx = (t + 1) * currSize;
        //Доходя до краев в каждом случае пересчитваем умф от предыщих значений справа и слева
        if (begIdx == 0) {
            currU[currTIdx] = 0.0;
            if (endIdx == N)
                currU[currTIdx + currN] = 0.0;
            else
                currU[currTIdx + currN] = countRightPrev(currN, currSize, currU, right_prev, t);
        } else {
            if (endIdx == N) {
                currU[currTIdx + currN] = 0.0;
                currU[currTIdx] = countLeftPrev(currSize, currU, left_prev, t);
            } else {
                if (currN == 0)
                    currU[currTIdx] = 2.0 * (1.0 - L2) * currU[t * currSize]
                                      + L2 * (left_prev + right_prev)
                                      - currU[(t - 1) * currSize];
                else {
                    currU[currTIdx + currN] = countRightPrev(currN, currSize, currU, right_prev, t);
                    currU[currTIdx] = countLeftPrev(currSize, currU, left_prev, t);
                }
            }
        }

        for (int i = 1; i < currN; i++) {
            currU[currTIdx + i] = umf(currSize, currU, t, i);
        }

        MPI_Send(&currU[currTIdx], 1, MPI_DOUBLE, left, 0, comm_1D);
        MPI_Send(&currU[currTIdx + currN], 1, MPI_DOUBLE, right, 0, comm_1D);

        MPI_Recv(&left_prev, 1, MPI_DOUBLE, left, 0, comm_1D, &status1);
        MPI_Recv(&right_prev, 1, MPI_DOUBLE, right, 0, comm_1D, &status2);
    }


    double *fullTime = nullptr;
    double *uFull = nullptr;
    if (rank == 0) {
        uFull = new double[(Nt + 1) * size * matrixPart];
        fullTime = new double[size];
    }

    //Собираем кусочки
    MPI_Gather(&currU[0], Nt + 1, MPI_RAW,
               &uFull[0], Nt + 1, MPI_RAW, 0, comm_1D);

    time1 = MPI_Wtime() - time1;
    //Собираем время
    MPI_Gather(&time1, 1, MPI_DOUBLE,
               fullTime, 1, MPI_DOUBLE, 0, comm_1D);

    if (rank == 0) {

        ofstream fileOutput = ofstream("output.txt");

        fileOutput << "NxNT: " << fixed << N << "x" << Nt << "\n";

        auto maxTime = fullTime[0];
        for (int i = 0; i < size; i++) {
            if (fullTime[i] > maxTime)
                maxTime = fullTime[i];
        }
        fileOutput << "Time: " << maxTime << "\n";

        //выводим координаты x
        fileOutput << 0 << ",";
        for (int i = 0; i < N; i++) {
            fileOutput << i * h << ",";
        }
        fileOutput << 1 << "\n";

        for (int i = 0; i < Nt + 1; i++) {
            //координаты по T
            fileOutput << i * tau << ",";
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < matrixPart; k++) {
                    if (j == size - 1 && k == matrixPart - 1)
                        fileOutput << uFull[(j * (Nt + 1) + i) * matrixPart + k] << "\n";
                    else fileOutput << uFull[(j * (Nt + 1) + i) * matrixPart + k] << ",";
                }
            }
        }
        delete[] uFull;
        fileOutput.close();
    }

    //чистим память
    MPI_Type_free(&MPI_RAW);
    MPI_Comm_free(&comm_1D);
    MPI_Finalize();
    delete[] currU;
    return 0;
}
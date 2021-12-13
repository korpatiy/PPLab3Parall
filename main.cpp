#include <iostream>
#include <cmath>
#include <fstream>
#include "mpi.h"

using namespace std;

const int N = 10;
const double a = 1.0;
const double h = a / N;
const double tau = h * h / 2 / 2;
const double T = 0.5;
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

//функция по методичке
double funcExample(double x) {
    if (x < 0.5) return 2.0 * x;
    return 2.0 - 2.0 * x;
}

//Подпрограмма по методичке
/*double *pp(double tau, double h, const double *y1, const double *y2) {
  auto *y3 = new double[N + 1];
  y3[0] = 0;
  y3[N] = 0;
  for (int i = 1; i < N; i++) {
    //была найдена ошибка -> во втором слагаемом +
    y3[i] = 2 * (1 - L2) * y2[i] + L2 * (y2[i + 1] - y2[i - 1]) - y1[i];
  }
  return y3;
}*/

int main(int argc, char **argv) {
    int rank, size;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double time = MPI_Wtime();

    MPI_Comm comm_1D;
    int dims[1] = {size};
    int periods[1] = {0};
    MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, 0, &comm_1D);

    //Координаты соседей
    int left;
    int right;
    MPI_Cart_shift(comm_1D, 0, 1, &left, &right);

    auto *u = new double[(Nt + 1) * (N + 1)];

    //начальные и краевые условия
    u[N + 1] = u[N + N + 1] = u[N] = u[0] = 0.0;
    for (int i = 1; i < N; i++) {
        //x от 0 до 1 в i * h
        u[0 * (N + 1) + i] = func(i * h); //y1
        u[1 * (N + 1) + i] = u[0 * (N + 1) + i] + tau * g(i); //y2
    }

    //Идем по времени от 0 до T. y3 замена функции pp
    for (int t = 1; t < Nt; t++) {
        //Производим обмен между граничными процессами
        if (rank % 2 == 0) {
            MPI_Send(&u[1 + (N + 1)], 1, MPI_DOUBLE, left, 0, comm_1D);
            MPI_Send(&u[N + (N + 1)], 1, MPI_DOUBLE, right, 0, comm_1D);

            MPI_Recv(&u[0 + (N + 1)], 1, MPI_DOUBLE, left, 0, comm_1D, &status);
            MPI_Recv(&u[N + 1 + (N + 1)], 1, MPI_DOUBLE, right, 0, comm_1D, &status);
        } else {
            MPI_Recv(&u[0 + (N + 1)], 1, MPI_DOUBLE, left, 0, comm_1D, &status);
            MPI_Recv(&u[N + 1 + (N + 1)], 1, MPI_DOUBLE, right, 0, comm_1D, &status);

            MPI_Send(&u[1 + (N + 1)], 1, MPI_DOUBLE, left, 0, comm_1D);
            MPI_Send(&u[N + 1 + (N + 1)], 1, MPI_DOUBLE, right, 0, comm_1D);
        }

        //Вычисление УМФ -> подпрограмма pp по факту
        u[(t + 1) * (N + 1) + 0] = 0.0;
        u[(t + 1) * (N + 1) + N] = 0.0;
        for (int i = 1; i < N; i++) {
            u[(t + 1) * (N + 1) + i] =
                    2.0 * (1.0 - L2) * u[t * (N + 1) + i] + L2 * (u[t * (N + 1) + i + 1] + u[t * (N + 1) + i - 1])
                    - u[(t - 1) * (N + 1) + i];
        }
    }

    //Собираем все
    auto *u_full = new double[4 * (Nt + 1) * (N + 1)]{0};
    MPI_Gather(u, (Nt + 1) * (N + 1), MPI_DOUBLE, u_full, (Nt + 1) * (N + 1), MPI_DOUBLE, 0, comm_1D);

    time = MPI_Wtime() - time;
    auto *full_time = new double[size];
    MPI_Gather(&time, 1, MPI_DOUBLE, full_time, 1, MPI_DOUBLE, 0, comm_1D);

    if (rank == 0) {
        double max_time = full_time[0];
        for (int i = 0; i < size; i++) {
            if (full_time[i] > max_time)
                max_time = full_time[i];
        }

        ofstream fileOutput = ofstream("output.txt");
        fileOutput << "Time: " << max_time * 1000 << "\n";

        //выводим координаты x
        fileOutput << 0 << ",";
        for (int i = 0; i < N; i++) {
            fileOutput << i * h << ",";
        }
        fileOutput << 1 << "\n";

        for (int i = 0; i <= Nt; i++) {
            //координаты по T
            fileOutput << i * tau << ",";
            for (int j = 0; j < N; j++) {
                fileOutput << u_full[i * (N + 1) + j] << ",";
            }
            fileOutput << u_full[i * (N + 1) + N];
            fileOutput << "\n";
        }
        fileOutput.close();
    }

    //чистим память
    MPI_Barrier(comm_1D);
    MPI_Finalize();
    delete[] u;
    delete[] u_full;
    delete[] full_time;
    return 0;
}
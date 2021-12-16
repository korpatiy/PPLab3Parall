#include <iostream>
#include <cmath>
#include <fstream>
#include "mpi.h"

using namespace std;

const int N = 11;
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
  MPI_Status status;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  double time = MPI_Wtime();

  int dims[1] = {0};
  MPI_Dims_create(size, 1, dims);
  int periods[1] = {0};

  MPI_Comm comm_1D;
  MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, true, &comm_1D);

  //Координаты соседей
  int left;
  int right;
  //Нахождение соседей слева и справа
  MPI_Cart_shift(comm_1D, 0, 1, &left, &right);

  MPI_Comm_rank(comm_1D, &rank);

  //Нарезаем часть, которую будет обрабатывать каждый процессор
  auto matrixPart = (Nt + 1) / size;

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
  if (begIdx == 0)
    currU[currSize] = currU[0] = 0.0;

  if (endIdx == N)
    currU[currSize + currN] = currU[currN] = 0.0;

  for (int i = 1; i <= currN; i++) {
    //x от 0 до 1 в i * h
    currU[i] = func((begIdx + i) * h); //y1
    currU[currSize + i] = currU[i] + tau * g(begIdx + i); //y2
  }

  double left_prev, right_prev;
  MPI_Send(&currU[1 * currSize + 0], 1, MPI_DOUBLE, left, 0, comm_1D);
  MPI_Send(&currU[1 * currSize + currN], 1, MPI_DOUBLE, right, 0, comm_1D);
  MPI_Recv(&left_prev, 1, MPI_DOUBLE, left, 0, comm_1D, &status);
  MPI_Recv(&right_prev, 1, MPI_DOUBLE, right, 0, comm_1D, &status);

  //Идем по времени от 0 до T.
  for (int t = 1; t < Nt; t++) {
    if (begIdx == 0) {
      currU[(t + 1) * currSize] = 0.0;
      if (endIdx == N)
        currU[(t + 1) * currSize + currN] = 0.0;
      else
        currU[(t + 1) * currSize + currN] = countRightPrev(currN, currSize, currU, right_prev, t);
    } else {
      if (endIdx == N) {
        currU[(t + 1) * currSize + currN] = 0.0;
        currU[(t + 1) * currSize] = countLeftPrev(currSize, currU, left_prev, t);
      } else {
        if (currN == 0)
          currU[(t + 1) * currSize] = 2.0 * (1.0 - L2) * currU[t * currSize]
              + L2 * (left_prev + right_prev)
              - currU[(t - 1) * currSize];
        else {
          currU[(t + 1) * currSize + currN] = countRightPrev(currN, currSize, currU, right_prev, t);
          currU[(t + 1) * currSize] = countLeftPrev(currSize, currU, left_prev, t);
        }
      }
    }

    for (int i = 1; i < currN; i++) {
      currU[(t + 1) * currSize + i] = umf(currSize, currU, t, i);
    }

    MPI_Send(&currU[(t + 1) * currSize], 1, MPI_DOUBLE, left, 0, comm_1D);
    MPI_Send(&currU[(t + 1) * currSize + currN], 1, MPI_DOUBLE, right, 0, comm_1D);

    MPI_Recv(&left_prev, 1, MPI_DOUBLE, left, 0, comm_1D, &status);
    MPI_Recv(&right_prev, 1, MPI_DOUBLE, right, 0, comm_1D, &status);
  }

  auto *uFull = new double[(Nt + 1) * size * matrixPart];

  //Собираем
  MPI_Gather(&currU[0], Nt + 1, MPI_RAW,
             &uFull[0], Nt + 1, MPI_RAW, 0, comm_1D);

  if (rank == 0) {

    ofstream fileOutput = ofstream("output.txt");
    //fileOutput << "Time: " << max_time * 1000 << "\n";

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
        fileOutput << uFull[i * (N + 1) + j] << ",";
      }
      fileOutput << uFull[i * (N + 1) + N];
      fileOutput << "\n";
    }
    fileOutput.close();
  }

  //чистим память
  MPI_Barrier(comm_1D);
  MPI_Finalize();
  delete[] currU;
  delete[] uFull;
  return 0;
}
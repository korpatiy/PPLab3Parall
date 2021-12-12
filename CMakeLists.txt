cmake_minimum_required(VERSION 3.20)
project(PPLab3Parall)

set(CMAKE_CXX_STANDARD 14)
include_directories("/opt/homebrew/include/open-mpi")
set(CMAKE_CXX_COMPILER mpic++)

add_executable(PPLab3Parall main.cpp)

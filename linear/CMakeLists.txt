cmake_minimum_required(VERSION 3.17)
project(ml_lab2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(
    ml_lab2
        main.cpp
        vec.hpp
)


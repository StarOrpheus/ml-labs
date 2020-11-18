cmake_minimum_required(VERSION 3.17)
project(ml_lab1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_subdirectory(3rd-party)

include_directories(3rd-party/csv-parser-2.1.0/include)

add_executable(ml_lab1 main.cpp)
target_link_libraries(ml_lab1 csv)
cmake_minimum_required(VERSION 2.8)

project(klt)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(klt klt.cpp)
target_link_libraries(klt ${OpenCV_LIBS})

cmake_minimum_required(VERSION 3.0.2)

project(example)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -g")

add_executable(example main.cpp Template/CustomCV.cpp)
target_link_libraries(example ${OpenCV_LIBS})
install(TARGETS example RUNTIME DESTINATION bin)

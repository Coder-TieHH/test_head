cmake_minimum_required(VERSION 3.9)
set(CMAKE_CXX_STANDARD 14)
project(test_head)

set(CMAKE_BUILD_TYPE DEBUG) # Debug模式 

set(ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})
message(STATUS "Current build dir:" ${CMAKE_CURRENT_LIST_DIR})

#set(OpenCV_STATIC ON) 
#set(OpenCV_DIR "/opt/homebrew/Cellar/opencv@3/3.4.16_1/share/OpenCV")  
find_package(OpenCV REQUIRED)

# 引入外部库的方式1, 制定外部库的搜索路径 
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib) 

# 引入外部库的方式2 以导入外部库的方式(不会重新编译)重新构建一个ncnn库, 然后可执行程序链接到该库
# add_library(ncnn STATIC IMPORTED) # 导入 ncnn dynamic lib 
# set_target_properties(ncnn PROPERTIES IMPORTED_LOCATION  ${ROOT_DIR}/lib/libncnn.a)

FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND) 
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS"${CMAKE_EXE_LINKER_FLAGS}${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR} 
    ${CMAKE_CURRENT_SOURCE_DIR}/common/
)

add_executable(test_head src/test_head.cpp src/tengine_operations.c)


target_link_libraries(
      test_head
    tengine-lite
    ${OpenCV_LIBS}
)

# target_link_libraries(nanotrack_demo /home/rpdzkj/Desktop/NanoTrack-master/tengine_nanotrack/lib/libtengine-lite.so ${OpenCV_LIBS}) 

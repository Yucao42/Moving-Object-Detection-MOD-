cmake_minimum_required(VERSION 2.8)
project(MOD)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# OpenCV 
find_package( OpenCV REQUIRED )
# 
include_directories( 
 include
${OpenCV_INCLUDE_DIRS} 
)

#add_executable( gms_match_demo ./src/demo_video.cpp )
add_executable(MOD ./example/demo.cpp include/header.hpp src/mod.cpp)
target_link_libraries( MOD ${OpenCV_LIBS} )

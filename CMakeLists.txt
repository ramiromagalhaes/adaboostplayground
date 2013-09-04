cmake_minimum_required(VERSION 2.8)

project(adaboostplayground)

# Debug and release configurations
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -D_DEBUG -Wextra -Wall")
set(CMAKE_CSS_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

find_package( OpenCV REQUIRED COMPONENTS core imgproc )
#find_package( Boost REQUIRED COMPONENTS filesystem system serialization )

# Setting the main program
set(source_files
    Common.h
    StrongHypothesis.h
    WeakLearner.h ResamplingWeakLearner.h ReweightingWeakLearner.h
    Adaboost.h
    Main.cpp)
add_executable( adaboostplayground ${source_files} )
target_link_libraries( adaboostplayground ${OpenCV_LIBS} )

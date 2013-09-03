cmake_minimum_required(VERSION 2.8)

project(adaboostplayground)

# Debug and release configurations
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -D_DEBUG -Wextra -Wall")
set(CMAKE_CSS_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")


# Setting the main program
set(source_files Main.cpp
    Adaboost.h
    Common.h
    StrongHypothesis.h
    WeakHypothesis.h
    WeakLearner.h)
add_executable( adaboostplayground ${source_files} )

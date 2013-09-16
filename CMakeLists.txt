cmake_minimum_required(VERSION 2.8)

project(adaboostplayground)

# Debug and release configurations
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -D_DEBUG -Wextra -Wall -pedantic")
set(CMAKE_CSS_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

find_package( OpenCV REQUIRED COMPONENTS core imgproc highgui)

# Depends on my haarcommon library
include_directories( /home/ramiro/workspace/haarcommon/ )
link_directories( /home/ramiro/workspace/haarcommon-build/ )

# Setting the main program
set(source_files
    common.h
    labeledexample.h
    stronghypothesis.h
    adaboost.h
    main.cpp
    haarclassifier.h
    haarclassifier.cpp
    sampleextractor.h
    sampleextractor.cpp)
add_executable( adaboostplayground ${source_files} )
target_link_libraries( adaboostplayground haarcommon ${OpenCV_LIBS} ${Boost_LIBRARIES} tbb)

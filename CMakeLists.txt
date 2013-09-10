cmake_minimum_required(VERSION 2.8)

project(adaboostplayground)

# Debug and release configurations
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -D_DEBUG -Wextra -Wall -pedantic")
set(CMAKE_CSS_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

find_package( OpenCV REQUIRED COMPONENTS core imgproc highgui)

# Depends on OpenImageIO
include_directories(/home/ramiro/workspace/oiio/dist/linux64.debug/include)
link_directories(/home/ramiro/workspace/oiio/dist/linux64.debug/lib)

# Setting the main program
set(source_files
    Common.h
    StrongHypothesis.h
    Adaboost.h
    Main.cpp
    Haarwavelet.h Haarwavelet.cpp
    Detector.h Detector.cpp
    Haarwaveletutilities.h Haarwaveletutilities.cpp
    dataprovider.h dataprovider.cpp )
add_executable( adaboostplayground ${source_files} )
target_link_libraries( adaboostplayground ${OpenCV_LIBS} ${Boost_LIBRARIES} OpenImageIO)

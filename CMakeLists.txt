cmake_minimum_required(VERSION 2.8)

project(adaboostplayground)

# Debug and release configurations
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -D_DEBUG -Wextra -Wall -pedantic")
set(CMAKE_CSS_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

find_package( OpenCV REQUIRED COMPONENTS core imgproc highgui)

# Depends on OpenImageIO and on my haarcommon library
include_directories(/home/ramiro/workspace/oiio/dist/linux64.debug/include
                    /home/ramiro/workspace/haarcommon/ )
link_directories(/home/ramiro/workspace/oiio/dist/linux64.debug/lib
                 /home/ramiro/workspace/haarcommon-build/)

set(dataprovider_files dataprovider.h dataprovider.cpp sampleextractor.h sampleextractor.cpp)
add_library( dataprovider STATIC ${dataprovider_files} )
target_link_libraries( dataprovider ${OpenCV_LIBS} OpenImageIO)

# Setting the main program
set(source_files
    common.h
    labeledexample.h
    stronghypothesis.h
    adaboost.h
    main.cpp
    haarclassifier.h haarclassifier.cpp )
add_executable( adaboostplayground ${source_files} )
target_link_libraries( adaboostplayground haarcommon dataprovider ${OpenCV_LIBS} ${Boost_LIBRARIES} tbb)

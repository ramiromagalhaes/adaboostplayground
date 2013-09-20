cmake_minimum_required(VERSION 2.8)

project(adaboostplayground)

# Debug and release configurations
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -D_DEBUG -Wextra -Wall -pedantic")
set(CMAKE_CSS_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

find_package( OpenCV REQUIRED COMPONENTS core imgproc highgui)

# Depends on my haarcommon library
include_directories( /home/ramiro/workspace/haarcommon/ )
link_directories( /home/ramiro/workspace/haarcommon-build/ )

# Reference to external lib
# You can define two import-locations: one for debug and one for release.
add_library( haarcommon-release SHARED IMPORTED )
set_target_properties( haarcommon-release PROPERTIES IMPORTED_LOCATION /home/ramiro/workspace/haarcommon-build/libhaarcommon.a )
add_library( haarcommon-debug SHARED IMPORTED )
set_target_properties( haarcommon-debug PROPERTIES IMPORTED_LOCATION /home/ramiro/workspace/haarcommon-build-debug/libhaarcommon.a )



# Setting the main program
set(common_source_files
    common.h
    labeledexample.h
    stronghypothesis.h
    adaboost.h
    haarclassifier.h
    haarclassifier.cpp
    sampleextractor.h
    sampleextractor.cpp )
add_executable( trainclassifier trainclassifier.cpp template_trainclassifier.h ${common_source_files} )
target_link_libraries( trainclassifier debug     haarcommon-debug   ${OpenCV_LIBS} ${Boost_LIBRARIES} tbb)
target_link_libraries( trainclassifier optimized haarcommon-release ${OpenCV_LIBS} ${Boost_LIBRARIES} tbb)

add_executable( train_my_classifier train_my_classifier.cpp template_trainclassifier.h ${common_source_files} )
target_link_libraries( train_my_classifier debug     haarcommon-debug   ${OpenCV_LIBS} ${Boost_LIBRARIES} tbb)
target_link_libraries( train_my_classifier optimized haarcommon-release ${OpenCV_LIBS} ${Boost_LIBRARIES} tbb)

add_executable( testclassifier testclassifier.cpp template_testclassifier.h ${common_source_files} )
target_link_libraries( testclassifier debug     haarcommon-debug   ${OpenCV_LIBS} ${Boost_LIBRARIES} tbb)
target_link_libraries( testclassifier optimized haarcommon-release ${OpenCV_LIBS} ${Boost_LIBRARIES} tbb)

add_executable( test_my_classifier test_my_classifier.cpp template_testclassifier.h ${common_source_files} )
target_link_libraries( test_my_classifier debug     haarcommon-debug   ${OpenCV_LIBS} ${Boost_LIBRARIES} tbb)
target_link_libraries( test_my_classifier optimized haarcommon-release ${OpenCV_LIBS} ${Boost_LIBRARIES} tbb)

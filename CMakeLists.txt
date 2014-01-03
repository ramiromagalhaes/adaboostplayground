project( adaboostplayground )
cmake_minimum_required( VERSION 2.8 )

# Debug and release configurations
set( CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -D_DEBUG -Wextra -Wall -pedantic" )
set( CMAKE_CSS_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall" )

# This is where all binaries should be placed
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} )

# Include OpenCV and Boost libraries
find_package( OpenCV REQUIRED COMPONENTS core imgproc highgui )
find_package( Boost REQUIRED COMPONENTS filesystem system )
# TODO What about Intel TBB?

# The common header files found in this project
include_directories( common )

# Depends on my haarcommon library
include_directories( /home/ramiro/workspace/haarcommon/src/ )
add_library( haarcommon-release SHARED IMPORTED )
add_library( haarcommon-debug   SHARED IMPORTED )
set_target_properties( haarcommon-release PROPERTIES IMPORTED_LOCATION /home/ramiro/workspace/haarcommon-build/src/libhaarcommon.so )
set_target_properties( haarcommon-debug   PROPERTIES IMPORTED_LOCATION /home/ramiro/workspace/haarcommon-build-debug/src/libhaarcommon.so )

# The subprojects
add_subdirectory(common)
add_subdirectory(train)
add_subdirectory(test)

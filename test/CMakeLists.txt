# TESTING tools build file
set(test_source_files
    scanner.h
    testdatabase.h
    testdatabase.cpp
    template_testclassifier.h)

#TESTING Programs
add_executable( test_vj_classifier     test_vj_classifier.cpp     ${test_source_files} )
target_link_libraries( test_vj_classifier debug     haarcommon-debug   tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( test_vj_classifier optimized haarcommon-release tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )

add_executable( test_band_classifier     test_band_classifier.cpp     ${test_source_files} )
target_link_libraries( test_band_classifier debug     haarcommon-debug   tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( test_band_classifier optimized haarcommon-release tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )

add_executable( test_pavani_classifier test_pavani_classifier.cpp ${test_source_files} )
target_link_libraries( test_pavani_classifier debug     haarcommon-debug   tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( test_pavani_classifier optimized haarcommon-release tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )

add_executable( test_normhist_classifier     test_normhist_classifier.cpp     ${test_source_files} )
target_link_libraries( test_normhist_classifier debug     haarcommon-debug   tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( test_normhist_classifier optimized haarcommon-release tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )

add_executable( test_adhikari_classifier     test_adhikari_classifier.cpp     ${test_source_files} )
target_link_libraries( test_adhikari_classifier debug     haarcommon-debug   tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( test_adhikari_classifier optimized haarcommon-release tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )

add_executable( test_rasolzadeh_classifier     test_rasolzadeh_classifier.cpp     ${test_source_files} )
target_link_libraries( test_rasolzadeh_classifier debug     haarcommon-debug   tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( test_rasolzadeh_classifier optimized haarcommon-release tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )

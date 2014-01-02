# TRAINING tools build file
set(train_source_files
    weaklearner.h
    sampleextractor.h
    sampleextractor.cpp
    adaboost.h
    template_trainclassifier.h)

#TRAINNING Programs
add_executable( train_vj_classifier train_vj_classifier.cpp         ${train_source_files} )
target_link_libraries( train_vj_classifier debug         haarcommon-debug   tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( train_vj_classifier optimized     haarcommon-release tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )

add_executable( train_my_classifier train_my_classifier.cpp         ${train_source_files} )
target_link_libraries( train_my_classifier debug         haarcommon-debug   tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( train_my_classifier optimized     haarcommon-release tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )

add_executable( train_pavani_classifier train_pavani_classifier.cpp ${train_source_files} )
target_link_libraries( train_pavani_classifier debug     haarcommon-debug   tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( train_pavani_classifier optimized haarcommon-release tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )


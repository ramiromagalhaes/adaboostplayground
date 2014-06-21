# TRAINING tools build file
set(headers
    progresscallback.h
    weaklearner.h
    sampleextractor.h
    adaboost.h
    template_trainclassifier.h)

set(source
    progresscallback.cpp
    sampleextractor.cpp)

set(train_program ${headers} ${source})

#TRAINNING Programs
add_executable( train_vj_classifier train_vj_classifier.cpp         ${train_program} )
target_link_libraries( train_vj_classifier debug            haarcommon-debug   tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( train_vj_classifier optimized        haarcommon-release tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )

add_executable( train_band_classifier train_band_classifier.cpp         ${train_program} )
target_link_libraries( train_band_classifier debug          haarcommon-debug   tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( train_band_classifier optimized      haarcommon-release tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )

add_executable( train_pavani_classifier train_pavani_classifier.cpp ${train_program} )
target_link_libraries( train_pavani_classifier debug        haarcommon-debug   tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( train_pavani_classifier optimized    haarcommon-release tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )

add_executable( train_my2nd_classifier train_my2nd_classifier.cpp   ${train_program} )
target_link_libraries( train_my2nd_classifier debug         haarcommon-debug   tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( train_my2nd_classifier optimized     haarcommon-release tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )

add_executable( train_my3rd_classifier train_my3rd_classifier.cpp   ${train_program} )
target_link_libraries( train_my3rd_classifier debug         haarcommon-debug   tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( train_my3rd_classifier optimized     haarcommon-release tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )

add_executable( train_adhikari_classifier train_adhikari_classifier.cpp   ${train_program} )
target_link_libraries( train_adhikari_classifier debug      haarcommon-debug   tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( train_adhikari_classifier optimized  haarcommon-release tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )

add_executable( train_histhist_classifier train_histhist_classifier.cpp         ${train_program} )
target_link_libraries( train_histhist_classifier debug      haarcommon-debug   tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( train_histhist_classifier optimized  haarcommon-release tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )

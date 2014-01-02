#ifndef TEMPLATE_TESTCLASSIFIER_H
#define TEMPLATE_TESTCLASSIFIER_H

#include <vector>
#include <iostream>

#include <fstream>
#include <sstream>
#include <tbb/tbb.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/unordered_map.hpp>
#include <boost/filesystem.hpp>

#include "common.h"
#include "stronghypothesis.h"
#include "scanner.h"
#include "weakhypothesis.h"
#include "sampleextractor.h"



/**
 * Arguments:
 *     positivesIndexFile
 *     positivesImageFile
 *     negativesIndexFile
 *     negativesImageFile
 *     strongHypothesisInputFile
 */
int main(int, char **argv) {
    const std::string imageFile = argv[1];
    const std::string strongHypothesisFile = argv[2];



    StrongHypothesis<MyHaarClassifier> strongHypothesis;
    {
        std::ifstream in(strongHypothesisFile.c_str());
        if ( !in.is_open() )
        {
            return 7;
        }
        if ( !strongHypothesis.read(in) )
        {
            return 11;
        }

        std::cout << "Loaded strong classifier." << std::endl;
    }



    cv::Mat image = cv::imread(imageFile, cv::DataType<unsigned char>::type);
    if ( !image.data )
    {
        return 1;
    }

    std::vector<cv::Rect> detections;
    Scanner<MyHaarClassifier> scanner(&strongHypothesis);
    scanner.scan(image, detections);

    cv::Mat outputImage(image.rows, image.cols, CV_8UC3); //don't know how to use datatype here
    image.copyTo(outputImage);

    for(std::vector<cv::Rect>::iterator detection = detections.begin(); detection != detections.end(); ++detection)
    {
        cv::rectangle(outputImage, *detection, cv::Scalar(255,0,0));
    }

    cv::imshow("Detection results", outputImage);
    cv::waitKey(0);

    return 0;
}


#endif // TEMPLATE_TESTCLASSIFIER_H

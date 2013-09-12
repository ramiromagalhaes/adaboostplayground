#ifndef LABELEDEXAMPLE_H
#define LABELEDEXAMPLE_H

#include "common.h"
#include <opencv2/imgproc/imgproc.hpp>



/**
 * Holds the sample and its classification.
 */
class LabeledExample {
public:
    cv::Mat integralSum;
    cv::Mat integralSquare;
    Classification label;

    LabeledExample() : integralSum(21, 21, CV_64F),
                       integralSquare(21, 21, CV_64F),
                       label(no) {}

    LabeledExample (const cv::Mat e, const Classification c) : integralSum(21, 21, CV_64F),
                                                               integralSquare(21, 21, CV_64F),
                                                               label(c)
    {
        updateIntegrals(e);
    }

    inline void updateIntegrals(const cv::Mat & image)
    {
        cv::integral(image, integralSum, integralSquare, CV_64F);
    }
};



/**
 * @brief LEContainer LabeledExample container type.
 */
typedef std::vector< LabeledExample > LEContainer;



#endif // LABELEDEXAMPLE_H

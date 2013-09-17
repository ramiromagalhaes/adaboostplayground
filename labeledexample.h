#ifndef LABELEDEXAMPLE_H
#define LABELEDEXAMPLE_H

#include <opencv2/imgproc/imgproc.hpp>

#include "common.h"



/**
 * Holds the sample and its classification.
 */
class LabeledExample {
public:
    cv::Mat integralSum;
    cv::Mat integralSquare;
    Classification label;

    LabeledExample() : integralSum(21, 21, cv::DataType<double>::type),
                       integralSquare(21, 21, cv::DataType<double>::type),
                       label(no) {}

    LabeledExample (const cv::Mat e, const Classification c) : integralSum(21, 21, cv::DataType<double>::type),
                                                               integralSquare(21, 21, cv::DataType<double>::type),
                                                               label(c)
    {
        updateIntegrals(e);
    }

    inline void updateIntegrals(const cv::Mat & image)
    {
        cv::integral(image, integralSum, integralSquare, cv::DataType<double>::type);
        if (!integralSum.data || !integralSquare.data)
        {
            throw 137;
        }
    }
};



/**
 * @brief LEContainer LabeledExample container type.
 */
typedef std::vector< LabeledExample > LEContainer;



#endif // LABELEDEXAMPLE_H

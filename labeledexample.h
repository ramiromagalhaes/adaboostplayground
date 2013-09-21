#ifndef LABELEDEXAMPLE_H
#define LABELEDEXAMPLE_H

#include <opencv2/imgproc/imgproc.hpp>

#include "common.h"



class Example
{
private:
    cv::Mat integralSum;
    cv::Mat integralSquare;

    void updateIntegrals(const cv::Mat & image)
    {
        cv::integral(image, integralSum, integralSquare, cv::DataType<double>::type);
        if (!integralSum.data || !integralSquare.data)
        {
            throw 137;
        }
    }

public:
    Example() : integralSum   (21, 21, cv::DataType<double>::type),
                integralSquare(21, 21, cv::DataType<double>::type) {}

    Example(const cv::Mat & e) : integralSum   (21, 21, cv::DataType<double>::type),
                                 integralSquare(21, 21, cv::DataType<double>::type)
    {
        updateIntegrals(e);
    }

    Example(const cv::Mat & integralSum_, const cv::Mat & integralSquare_) : integralSum(integralSum_),
                                                                             integralSquare(integralSquare_)
    {
        if (!integralSum.data || !integralSquare.data || integralSum.size != integralSquare.size) //TODO should size also be 20?
        {
            throw 141;
        }
    }

    cv::Mat getIntegralSum() const
    {
        return integralSum;
    }

    cv::Mat getIntegralSquare() const
    {
        return integralSquare;
    }
};



/**
 * Holds the sample and its classification.
 */
class LabeledExample : public Example {
private:
    Classification label;

public:
    LabeledExample() : Example(),
                       label(no) {}

    LabeledExample (const cv::Mat & e, const Classification c) : Example(e),
                                                                 label(c) {}

    Classification getLabel() const
    {
        return label;
    }
};



/**
 * @brief LEContainer LabeledExample container type.
 */
typedef std::vector< LabeledExample > LEContainer;



#endif // LABELEDEXAMPLE_H

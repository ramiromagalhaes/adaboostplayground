#ifndef CLASSIFICATION_H_
#define CLASSIFICATION_H_

#include <opencv2/imgproc/imgproc.hpp>


/**
 * @brief weight_type Weighs used for training should adopt this type.
 *                    We use float as a default.
 */
typedef float weight_type;



/**
 * @brief WeightVector A vector to hold instances of weight_type.
 */
typedef std::vector<weight_type> WeightVector;



/**
 * @brief The Classification enum holds default values for the binary classification case.
 */
enum Classification {
    no = -1, yes = 1
};



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



#endif /* CLASSIFICATION_H_ */

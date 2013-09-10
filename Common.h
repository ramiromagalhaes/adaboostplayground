#ifndef CLASSIFICATION_H_
#define CLASSIFICATION_H_

#include <opencv2/core/core.hpp>



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
    cv::Mat example;
    Classification label;

    LabeledExample() : example(20, 20, CV_8UC1),
                       label(no) {}

    LabeledExample (const cv::Mat e, const Classification c) : example(e),
                                                               label(c) {}
};


/**
 * @brief LEContainer LabeledExample container type.
 */
typedef std::vector< LabeledExample > LEContainer;



#endif /* CLASSIFICATION_H_ */

#ifndef CLASSIFICATION_H_
#define CLASSIFICATION_H_

#include <opencv2/core.hpp>


/**
 * @brief weight_type Weighs used for training should adopt this type.
 *                    We use float as a default.
 */
typedef float weight_type;



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

    LabeledExample() {
        label = no;
    }

    LabeledExample (const cv::Mat e, const Classification c) : example(e), label(c) {
    }
};



/**
 * @brief The WeakHypothesis represents a weak hypothesis.
 */
class WeakHypothesis {
public:
    WeakHypothesis() { }
    virtual ~WeakHypothesis() { }
    virtual Classification classify(const cv::Mat &data) const =0;
};

#endif /* CLASSIFICATION_H_ */

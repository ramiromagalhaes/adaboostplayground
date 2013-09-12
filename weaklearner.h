#ifndef WEAKLEARNER_H_
#define WEAKLEARNER_H_

#include <vector>
#include "Common.h"

class WeakLearner {
public:
    WeakLearner() {} //note: intentional inline con/destructor. See http://stackoverflow.com/questions/644397/c-class-with-template-cannot-find-its-constructor

    virtual ~WeakLearner() {} //WeakLearner implementations should hold the data

    /**
     * @brief learn Trains a weak classifier that produces the smallest classification error in the training data set.
     * @param training_set The samples used to train the weak classifier.
     * @param weighted_errors The weighted errors of each training sample.
     * @param hypothesis The hypothesis collection the WeakLearner will train with.
     * @param weighted_error Output parameter. The classification error associated with the returned weak classifier.
     * @return A pointer to the WeakHypothesis that "minimalizes" the classification error.
     */
    virtual WeakHypothesis const * const learn(
        const LEContainer &training_set,
        const std::vector < weight_type > &weight_distribution,
        const std::vector < WeakHypothesis * > hypothesis,
        weight_type & weighted_error) =0;
};

#endif /* WEAKLEARNER_H_ */

#ifndef REWEIGHTINGWEAKLEARNER_H_
#define REWEIGHTINGWEAKLEARNER_H_

#include "WeakLearner.h"

class ReweightingWeakLearner : public WeakLearner {
public:
    ReweightingWeakLearner() {} //note: intentional inline con/destructor. See http://stackoverflow.com/questions/644397/c-class-with-template-cannot-find-its-constructor

    ~ReweightingWeakLearner() {} //WeakLearner implementations should hold the data

    /**
     * @brief learn Trains a weak classifier that produces the smallest classification error in the training data set.
     * @param training_set The samples used to train the weak classifier.
     * @param weighted_errors The weighted errors of each training sample.
     * @param hypothesis The hypothesis collection the WeakLearner will train with.
     * @param weighted_error Output parameter. The classification error associated with the returned weak classifier.
     * @return A pointer to the WeakHypothesis that "minimalizes" the classification error.
     */
    virtual WeakHypothesis const * const learn(
        const std::vector < LabeledExample > &training_set,
        const std::vector < weight_type > &weight_distribution,
        const std::vector < WeakHypothesis * > hypothesis,
        weight_type & weighted_error)
    {
        weight_type lowest_error = std::numeric_limits<weight_type>::max();
        WeakHypothesis const * best_hypothesis = 0;

        for (std::vector < WeakHypothesis * >::const_iterator it = hypothesis.begin(); it != hypothesis.end(); it++) {
            weight_type hypothesis_weighted_error = .0f;

            for (std::vector < LabeledExample >::size_type j = 0; j < training_set.size(); j++) {
                if ((*it)->classify(training_set[j].example) != training_set[j].label) {
                    hypothesis_weighted_error += weight_distribution[j];
                }
            }

            if (hypothesis_weighted_error < lowest_error) {
                lowest_error = hypothesis_weighted_error;
                best_hypothesis = *it;
            }
        }

        const weight_type maximum_weighted_error = 0.5f;
        if (lowest_error < maximum_weighted_error //checks for the weak learning assumption
                || best_hypothesis == 0) { //checks if some hypothesis was selected
            throw 10;
        }

        weighted_error = lowest_error;

        return best_hypothesis;
    }
};

#endif /* WEAKLEARNER_H_ */

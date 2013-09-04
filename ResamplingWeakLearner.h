#ifndef RESAMPLINGWEAKLEARNER_H_
#define RESAMPLINGWEAKLEARNER_H_

#include "WeakLearner.h"

class ResamplingWeakLearner : public WeakLearner {
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
    virtual WeakHypothesis * learn(
        const std::vector < LabeledExample > &training_set,
        const std::vector < weight_type > &weight_distribution,
        const std::vector < WeakHypothesis * > hypothesis,
        weight_type &weighted_error)
    {
        //TODO
    }

protected:

    /**
     * @brief binarySearchForSamples Run a binary search over a cumulative distribution and returns
     *                               the index of the key parameter located in the cumulative distribution
     *                               vector.
     * @param cumulative_distribution_weight The cumulative distribution vector. It is assumed that the values
     *                                       increase with the index.
     * @param key The value which index is to be found.
     * @return The index of key in cumulative_distribution_weight.
     */
    inline int binarySearchForSamples(
            std::vector<weight_type> &cumulative_distribution,
            weight_type key) {
        int first = 0;
        int last = cumulative_distribution.size() - 1;

        while (first <= last) {
            int mid = (first + last) / 2; //compute mid point.

            if (key <= cumulative_distribution[mid] && key >= cumulative_distribution[mid - 1]) {
                return mid; // found it. return position
            }

            if (key > cumulative_distribution[mid]) {
                first = mid + 1; // repeat search in top half.
            } else if (key < cumulative_distribution[mid]) {
                last = mid - 1; // repeat search in bottom half.
            }
        }

        throw 2; // failed to find key
    }



    /**
     * @brief init_cumulative_distribution Produces a cumulative distribution from probability distribution
     * @param sample_size
     * @param probability_distribution
     * @param cumulative_distribution
     */
    inline void init_cumulative_distribution(
            const int sample_size,
            const std::vector<weight_type> &probability_distribution,
            std::vector<weight_type> &cumulative_distribution) {
        for (int i = 0; i < sample_size; i++) {
            if (i) {
                cumulative_distribution[i] = cumulative_distribution[i - 1] + probability_distribution[i];
            } else {
                cumulative_distribution[i] = probability_distribution[i];
            }
        }

        cumulative_distribution[sample_size - 1] = 1;
    }



    /**
     * @brief resample Takes samples from training_set and, according to the probabilities found
     *                 in weight_distribution, produces a sample from it.
     * @param training_set The original full training set from which samples will be taken.
     * @param weight_distribution The weights of each element of training_set. This must be a
     *                            distribution, i.e., all elements must add up to 1.
     * @param sample_size The size of the resulting sample.
     * @param training_sample Output parameter. A collection of samples taken from training_set
     *                        according to weight_distribution.
     * @param training_sample_weight_distribution Output parameter. The weights of each element
     *                                            of training_sample.
     */
    void resample(
            const std::vector<LabeledExample> &training_set,
            const std::vector<weight_type> &weight_distribution,
            const unsigned int sample_size,
            std::vector<LabeledExample> &training_sample,
            std::vector<weight_type> &training_sample_weight_distribution) {

        //TODO assertion: all vectors must be of sample_size

        std::vector<weight_type> cumulative_distribution_weight(sample_size);
        init_cumulative_distribution(sample_size, weight_distribution, cumulative_distribution_weight);

        for (unsigned int i = 0; i < sample_size; ++i) {
            const weight_type random = (weight_type)rand() / (weight_type)RAND_MAX;
            const int index = binarySearchForSamples(cumulative_distribution_weight, random);

            training_sample[i] = training_set[index];
            training_sample_weight_distribution[i] = weight_distribution[index]; //TODO shouldn't this be a distribution too?
        }
    }



};

#endif /* WEAKLEARNER_H_ */

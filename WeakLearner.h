#ifndef WEAKLEARNER_H_
#define WEAKLEARNER_H_

#include <vector>
#include <utility>
#include <cstdlib>
#include "Common.h"
#include "WeakHypothesis.h"

template <typename dataType> class WeakLearner {
public:
    WeakLearner() {} //note: intentional inline con/destructor. See http://stackoverflow.com/questions/644397/c-class-with-template-cannot-find-its-constructor
    virtual ~WeakLearner() {} //WeakLearner implementations should hold the data

    virtual WeakHypothesis<dataType>* learn(
        const std::vector < LabeledExample <dataType> * > &training_set,
        const std::vector < weight_type > &weighted_errors,
        weight_type &weighted_error) =0;

protected:

    /**
     * @brief binarySearchForSamples Run a binary search over a cumulative distribution and returns
     *                               the index of the key parameter located in the cumulative distribution vector.
     * @param cumulative_distribution_weight The cumulative distribution vector. It is assumed that the values increase with the index.
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



    void resample(
            const unsigned int sample_size,
            const std::vector<weight_type> &weight_distribution,
            const std::vector < LabeledExample <dataType> * > &training_set,
            std::vector < LabeledExample <dataType> * > &training_sample,
            std::vector<weight_type> &training_sample_weight_distribution) {
        //TODO assertion: all vectors must be of sample_size

        std::vector<weight_type> cumulative_distribution_weight(sample_size);
        init_cumulative_distribution(sample_size, weight_distribution, cumulative_distribution_weight);

        for (unsigned int i = 0; i < sample_size; ++i) {
            const weight_type random = (weight_type)rand() / (weight_type)RAND_MAX;
            const int index = binarySearchForSamples(cumulative_distribution_weight, random);

            training_sample[i] = training_set[index];
            training_sample_weight_distribution[i] = weight_distribution[index];
        }
    }



};

#endif /* WEAKLEARNER_H_ */

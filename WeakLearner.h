/*
 * WeakLearner.h
 *
 *  Created on: 06/03/2013
 *      Author: ramiro
 */

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
	virtual ~WeakLearner() {
		//WeakLearner implementations should hold the data
	}

	virtual WeakHypothesis<dataType>* learn(
		const std::vector < LabeledExample <dataType> * > &training_set,
		const std::vector < double > &weighted_errors,
		double &weighted_error) =0;

protected:
	inline int binarySearchForSamples(
			std::vector<double> &cumulative_distribution_weight,
			double key) {
		int first = 0;
		int last = cumulative_distribution_weight.size() - 1;

		while (first <= last) {
			int mid = (first + last) / 2; //compute mid point.

			if (key <= cumulative_distribution_weight[mid] && key >= cumulative_distribution_weight[mid - 1]) {
				return mid; // found it. return position
			}

			if (key > cumulative_distribution_weight[mid]) {
				first = mid + 1; // repeat search in top half.
			} else if (key < cumulative_distribution_weight[mid]) {
				last = mid - 1; // repeat search in bottom half.
			}
		}

		throw 2; // failed to find key
	}



	inline void init_cumulative_distribution(
			const int sample_size,
			const std::vector<double> &distribution_weight,
			std::vector<double> &cumulative_distribution_weight) {
		for (int i = 0; i < sample_size; i++) {
			if (i) {
				cumulative_distribution_weight[i] = cumulative_distribution_weight[i - 1] + distribution_weight[i];
			} else {
				cumulative_distribution_weight[i] = distribution_weight[i];
			}
		}

		cumulative_distribution_weight[sample_size - 1] = 1;
	}



	void resample(
			const unsigned int sample_size,
			const std::vector<double> &weight_distribution,
			const std::vector < LabeledExample <dataType> * > &training_set,
			std::vector < LabeledExample <dataType> * > &training_sample,
			std::vector<double> &training_sample_weight_distribution) {
		//TODO assertion: all vectors must be of sample_size

		std::vector<double> cumulative_distribution_weight(sample_size);
		init_cumulative_distribution(sample_size, weight_distribution, cumulative_distribution_weight);

		for (unsigned int i = 0; i < sample_size; ++i) {
			const double random = (double)rand() / (double)RAND_MAX;
			const int index = binarySearchForSamples(cumulative_distribution_weight, random);

			training_sample[i] = training_set[index];
			training_sample_weight_distribution[i] = weight_distribution[index];
		}
	}



};

#endif /* WEAKLEARNER_H_ */

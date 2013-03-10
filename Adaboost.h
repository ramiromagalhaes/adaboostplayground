/*
 * Adaboost.h
 *
 *  Created on: 02/03/2013
 *      Author: ramiro
 */

#ifndef ADABOOST_H_
#define ADABOOST_H_

#include <vector>
#include <utility>
#include <math.h>
#include <numeric>
#include <algorithm>
#include <stdlib.h>
#include "Common.h"
#include "WeakLearner.h"
#include "WeakHypothesis.h"
#include "StrongHypothesis.h"




template<typename dataType> class Adaboost {
public:
	WeakLearner <dataType> *weak_learner;

	Adaboost(WeakLearner<dataType> *w_learner) {
		t= 0;
		weak_learner = w_learner;
	}

	virtual ~Adaboost() { }

	//TODO implement a way to allow different sampling strategies

private:
	int t; //iteration (epoch) counter

	void init_distribution(std::vector<double> &distribution) {
		for (std::vector<double>::iterator it = distribution.begin(); it != distribution.end(); ++it) {
			*it = 1.0 / distribution.size();
		}
	}

	void resample(
			const int sample_size,
			const std::vector<double> &distribution_weight,
			const std::vector < training_data <dataType> > &trainingData,
			std::vector < training_data <dataType> > &sample) {

		//TODO assertion: all vectors must be of sample_size

		std::vector<double> cumulative_distribution_weight(sample_size);
		init_cumulative_distribution(sample_size, distribution_weight, cumulative_distribution_weight);

		for (typename std::vector < training_data <dataType> >::iterator it = sample.begin(); it != sample.end(); ++it) {
			const double random = ((double)rand() / (double)RAND_MAX) * (double)sample_size;
			const int index = binarySearch(cumulative_distribution_weight, random);
			*it = trainingData[index];
		}
	}

	void init_cumulative_distribution(
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


	double evaluate_error(
			const WeakHypothesis<dataType> &weakHypothesis,
			const std::vector < training_data <dataType> > &trainingData) {
		return 0;
	}

	int binarySearch(
			std::vector<double> cumulative_distribution_weight,
			double key) {
		int first = 0;
		int last = cumulative_distribution_weight.size() - 1;

		while (first <= last) {
			int mid = (first + last) / 2; //compute mid point.

			if (key < cumulative_distribution_weight[mid] && key > cumulative_distribution_weight[mid - 1]) {
				return mid; // found it. return position
			}

			if (key > cumulative_distribution_weight[mid]) {
				first = mid + 1; // repeat search in top half.
			} else if (key < cumulative_distribution_weight[mid]) {
				last = mid - 1; // repeat search in bottom half.
			}
		}

		return -(first + 1);    // failed to find key
	}


	void update_distribution(
			const double alpha,
			WeakHypothesis<dataType> &current_weak_hypothesis,
			const std::vector < training_data <dataType> > &trainingData,
			std::vector<double> &distribution_weight) {
		const double normalizationFactor = std::accumulate(distribution_weight.begin(), distribution_weight.end(), 0);

		for (unsigned int i = 0; i < distribution_weight.size(); i++) {
			const Classification trainingResult = current_weak_hypothesis.test(trainingData[i].data);

			distribution_weight[i] = distribution_weight[i]
					* exp(-1.0 * alpha * trainingData[i].classification * trainingResult)
					/ normalizationFactor;
		}
	}

public:
	void train(
			const std::vector < training_data <dataType> > &trainingData,
			StrongHypothesis<dataType> strong_hypothesis) {
		//TODO iterations and stop criteria

		const int m = trainingData.size(); //the size of sample we'll take from the trainingData to train a WeakLearner each round
		std::vector<double> distribution_weight(m); //holds all weights elements in the last t
		init_distribution(distribution_weight);

	/*
		TODO Evaluate how these variables will be used.

		std::vector<double> alpha; //alpha subscript t where t is the vector's index
		std::vector<double> weightedError;
	*/

		//get a sample of the trainingData...
		std::vector < training_data <dataType> > sample(m);
		resample(m, distribution_weight, trainingData, sample);

		//train weak learner and get weak hypothesis
		WeakHypothesis<dataType> weak_hypothesis = weak_learner->learn(sample);

		//select h(t) to minimalize the weighted error
		const double weighted_error = evaluate_error(weak_hypothesis, trainingData);

		//choose alpha(t)
		const double alpha = log( (1.0 - weighted_error)/weighted_error ) / 2.0;

		//update the distribution
		update_distribution(alpha, weak_hypothesis, trainingData, distribution_weight);

		t++; //next training iteration

		//output final hypothesis
		strong_hypothesis.include(alpha, weak_hypothesis);
	}

};

#endif /* ADABOOST_H_ */

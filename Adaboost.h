/*
 * Adaboost.h
 *
 *  Created on: 02/03/2013
 *      Author: ramiro
 */

#ifndef ADABOOST_H_
#define ADABOOST_H_

#include <vector>
#include <math.h>
#include <numeric>
#include <algorithm>
#include "Common.h"
#include "WeakLearner.h"
#include "WeakHypothesis.h"
#include "StrongHypothesis.h"

#include <iostream>


template<typename dataType> class Adaboost {
public:
	Adaboost(WeakLearner<dataType> *w_learner) : t(0), weak_learner(w_learner) {
	}

	virtual ~Adaboost() { }

	//TODO implement means to allow different sampling strategies

private:
	int t; //iteration (epoch) counter
	WeakLearner <dataType> *weak_learner;



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
			const double random = ((double)rand() / (double)RAND_MAX);
			const int index = binarySearchForSamples(cumulative_distribution_weight, random);
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
			WeakHypothesis<dataType> const * const weakHypothesis,
			const std::vector < training_data <dataType> > &trainingData) {

		unsigned int wrong_conclusions = 0;

		for (typename std::vector < training_data<dataType> >::const_iterator itTrain = trainingData.begin(); itTrain != trainingData.end(); ++itTrain) {
			const training_data<dataType> train = *itTrain;

			if (weakHypothesis->classify(train.data) != train.classification) {
				wrong_conclusions++;
			}
		}

		return (double)wrong_conclusions / (double)trainingData.size();
	}



	int binarySearchForSamples(
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

		throw 2;
		//return -(first + 1);    // failed to find key
	}



	void update_distribution(
			const double alpha,
			WeakHypothesis<dataType> const * const current_weak_hypothesis,
			const std::vector < training_data <dataType> > &trainingData,
			std::vector<double> &distribution_weight) {
		const double normalizationFactor = std::accumulate(distribution_weight.begin(), distribution_weight.end(), 0);

		for (unsigned int i = 0; i < distribution_weight.size(); i++) {
			const Classification trainingResult = current_weak_hypothesis->classify(trainingData[i].data);

			distribution_weight[i] = distribution_weight[i]
					* exp(-1.0 * alpha * trainingData[i].classification * trainingResult)
					/ normalizationFactor;
		}
	}

public:
	void train(
			const std::vector < training_data <dataType> > &trainingData,
			StrongHypothesis<dataType> &strong_hypothesis) {

		const int m = trainingData.size(); //the size of sample we'll take from the trainingData to train a WeakLearner each round
		std::vector<double> distribution_weight(m); //holds all weights elements in the last t
		init_distribution(distribution_weight);

		//TODO How will I store the internal data?

		do { //TODO how will iterations and stop criteria work
			//Main Adaboost loop

			//get a sample of the trainingData...
			std::vector < training_data <dataType> > sample(m);
			resample(m, distribution_weight, trainingData, sample);

			//train weak learner and get weak hypothesis so that it minimalizes the weighted error
			WeakHypothesis<dataType> * weak_hypothesis = weak_learner->learn(sample);
			const double weighted_error = evaluate_error(weak_hypothesis, trainingData);

			//choose alpha(t)
			const double alpha = log( (1.0 - weighted_error)/weighted_error ) / 2.0;

			//update the distribution
			update_distribution(alpha, weak_hypothesis, trainingData, distribution_weight);

			t++; //next training iteration

			//output final hypothesis
			strong_hypothesis.insert(alpha, weak_hypothesis);
		} while (t < 3);
	}

};

#endif /* ADABOOST_H_ */

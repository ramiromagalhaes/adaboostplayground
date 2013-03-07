/*
 * Adaboost.cpp
 *
 *  Created on: 02/03/2013
 *      Author: ramiro
 */

#include "Adaboost.h"
#include "WeakHypothesis.h"
#include "WeakLearner.h"
#include <math.h>
#include <numeric>
#include <stdlib.h>



template<class dataType> Adaboost<dataType>::Adaboost() {
	t = 0;
}



template<class dataType> Adaboost<dataType>::~Adaboost() {}



template<class dataType> StrongHypothesis Adaboost<dataType>::train(const std::vector< std::pair<dataType, Classification> > &trainingData) {
	StrongHypothesis strongHypothesis; //this is what we wanna get in the end

	const int m = trainingData.size(); //the size of sample we'll take from the trainingData to train a WeakLearner each round
	std::vector<double> distribution_weight(m); //holds all weights elements in the last t
	init_distribution(distribution_weight);

	std::vector<double> alpha; //alpha subscript t where t is the vector's index
	std::vector<double> weightedError;

	//get a sample of the trainingData...
	const std::vector< std::pair<dataType, Classification> > sample;

	//train weak learner and get weak hypothesis
	WeakLearner<dataType> weakLearner;
	WeakHypothesis weakHypothesis = weakLearner.learn(sample);

	//select h(t) to minimalize the weighted error

	//choose alpha(t)

	//update the distribution

	t++; //next training iteration

	//output final hypothesis

	return strongHypothesis;
}

template<class dataType> void Adaboost<dataType>::init_distribution(std::vector<double> &distribution) {
	for (std::vector<double>::iterator it = distribution.begin();
			it != distribution.end(); ++it) {
		*it = 1.0 / distribution.size();
	}
}

/*
Hypothesis template<class dataType> Adaboost<dataType>::get_weak_hypothesis() {
	//TODO implement me (and purge the code below)
	Database data = Database();
	data.load();
	return data.hypothesisDatabase[0];
}



void template<class dataType> Adaboost<dataType>::choose_apha() {
	const double alpha_t = log((1.0 - weightedError[t]) / weightedError[t])
			/ 2.0;
	alpha.insert(alpha.end(), alpha_t);
}

void template<class dataType> Adaboost<dataType>::update_distribution(Hypothesis& currentWeakHypothesis) {
	const double normalizationFactor = std::accumulate(distribution.begin(),
			distribution.end(), 0);

	for (unsigned int i = 0; i < distribution.size(); i++) {
		const Point trainingInput = trainingData[i].first;
		const Classification trainingExpected = trainingData[i].second;
		const Classification trainingResult = currentWeakHypothesis.test(
				trainingInput);

		distribution[i] = distribution[i]
				* exp(-1.0 * alpha[t] * trainingExpected * trainingResult)
				/ normalizationFactor;
	}
}
*/

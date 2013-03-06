/*
 * Adaboost.cpp
 *
 *  Created on: 02/03/2013
 *      Author: ramiro
 */

#include "Adaboost.h"
#include "Data.h"
#include <math.h>
#include <numeric>
#include <stdlib.h>



Point::Point(double x, double y) {
	first = x;
	second = y;
}



Hypothesis::Hypothesis(Orientation o, double p) {
	orientation = o;
	position = p;
}



Classification Hypothesis::test(const Point input) {
	switch (orientation) {
		case vertical:
			//if input is to the left of the vertical bar, then return yes
			if (input.first <= position) {
				return yes;
			} else {
				return no;
			}
		case horizontal:
			//if input is above the horizontal bar, then return yes
			if (input.second >= position) {
				return yes;
			} else {
				return no;
			}
		default:
			throw "!!!";
	}
}



Adaboost::Adaboost() {
	t = 0;

	for (std::vector<double>::iterator it = distribution.begin();
			it != distribution.end(); ++it) {
		*it = 1.0 / distribution.size();
	}
}



Adaboost::~Adaboost() {}



void Adaboost::train() {
	//train weak learner
	//get weak hypothesis
	Hypothesis currentWeakHypothesis = get_weak_hypothesis();

	//select h(t) to minimalize the weighted error

	//choose alpha(t)
	choose_apha();

	//update the distribution
	update_distribution(currentWeakHypothesis);

	t++; //next training iteration

	//output final hypothesis
}



Hypothesis Adaboost::get_weak_hypothesis() {
	//TODO implement me (and purge the code below)
	Data data = Data();
	data.load();
	return data.hypothesisDatabase[0];
}



void Adaboost::choose_apha() {
	const double alpha_t = log((1.0 - weightedError[t]) / weightedError[t])
			/ 2.0;
	alpha.insert(alpha.end(), alpha_t);
}

void Adaboost::update_distribution(Hypothesis& currentWeakHypothesis) {
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

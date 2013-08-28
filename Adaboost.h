#ifndef ADABOOST_H_
#define ADABOOST_H_

#include <vector>
#include <math.h>
#include <numeric>
#include <algorithm>
#include <iostream>
#include "Common.h"
#include "WeakLearner.h"
#include "WeakHypothesis.h"
#include "StrongHypothesis.h"



template<typename dataType> class Adaboost {
public:
	Adaboost(WeakLearner<dataType> *w_learner) : t(0), weak_learner(w_learner) {
	}

	virtual ~Adaboost() { }

	//TODO implement means to allow different sampling strategies
	//TODO Store errors and historic data gathered through the iterations
	//TODO devise means to implement some flexible stop criteria

private:
	unsigned int t; //iteration (epoch) counter
	WeakLearner <dataType> *weak_learner;



	inline void init_distribution(std::vector<double> &distribution) {
		for (std::vector<double>::iterator it = distribution.begin(); it != distribution.end(); ++it) {
			*it = 1.0 / distribution.size();
		}
	}



	inline void update_distribution(
			const double alpha,
			WeakHypothesis<dataType> const * const current_weak_hypothesis,
			const std::vector < LabeledExample <dataType> * > &trainingData,
			std::vector<double> &distribution_weight) {

		double normalizationFactor = 0;

		for (std::vector<double>::size_type i = 0; i < distribution_weight.size(); i++) {
			const Classification trainingResult = current_weak_hypothesis->classify(trainingData[i]->example);

			distribution_weight[i] *= exp(-alpha * trainingData[i]->label * trainingResult);
			normalizationFactor += distribution_weight[i];
		}

		for (std::vector<double>::size_type i = 0; i < distribution_weight.size(); i++) {
			distribution_weight[i] /= normalizationFactor;
		}
	}



public:

    /**
     * @brief This method trains a strong classifier.
     * @param training_set A vector of LabeledExamples that will be used in training.
     * @param strong_hypothesis The object that will hold the strong classifier.
     * @param maximum_iterations The maximum iterations that this training will perform.
     */
    void train(
			const std::vector < LabeledExample <dataType> * > &training_set,
			StrongHypothesis<dataType> &strong_hypothesis,
			const unsigned int maximum_iterations) {

		std::vector<double> weight_distribution(training_set.size()); //holds all weight elements
		init_distribution(weight_distribution);

		do {//Main Adaboost loop

			//train weak learner and get weak hypothesis so that it minimalizes the weighted error
			double weighted_error = 0;
			WeakHypothesis<dataType> * weak_hypothesis =
					weak_learner->learn(training_set, weight_distribution, weighted_error);

			//choose alpha(t)
			const double alpha = log( (1.0 - weighted_error)/weighted_error ) / 2.0;

			//update the distribution
			update_distribution(alpha, weak_hypothesis, training_set, weight_distribution);

			//update the final hypothesis
			strong_hypothesis.insert(alpha, weak_hypothesis);

			t++; //next training iteration
		} while (t < maximum_iterations);
	}

};

#endif /* ADABOOST_H_ */

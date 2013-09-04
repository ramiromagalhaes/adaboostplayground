#ifndef ADABOOST_H_
#define ADABOOST_H_

#include <vector>
#include <cmath>
#include "Common.h"
#include "WeakLearner.h"
#include "ReweightingWeakLearner.h"
#include "StrongHypothesis.h"



class Adaboost {
public:
    Adaboost() : t(0),
                 weak_learner(new ReweightingWeakLearner()) {}

    Adaboost(WeakLearner * w_learner) : t(0),
                                        weak_learner(w_learner) {}

    virtual ~Adaboost() {
        delete weak_learner;
    }

    //TODO implement means to allow different sampling strategies
    //TODO Store errors and historic data gathered through the iterations
    //TODO devise means to implement some flexible stop criteria

protected:
    unsigned int t; /** iteration (epoch) counter */

    WeakLearner * weak_learner; /** the weak learner used in this Adaboost instance */



    /**
     * @brief init_distribution Initializes the sample weight distribution.
     * @param distribution
     */
    void init_distribution(std::vector<weight_type> &distribution) {
        for (std::vector<weight_type>::iterator it = distribution.begin(); it != distribution.end(); ++it) {
            *it = 1.0f / distribution.size();
        }
    }



    /**
     * @brief update_distribution Updates the weight distribution of the samples.
     * @param alpha
     * @param current_weak_hypothesis
     * @param trainingData
     * @param distribution_weight
     */
    void update_distribution(
            const weight_type alpha,
            WeakHypothesis const * const current_weak_hypothesis,
            const std::vector < LabeledExample > &trainingData,
            std::vector<weight_type> &distribution_weight) {

        weight_type normalizationFactor = 0;

        for (std::vector<weight_type>::size_type i = 0; i < distribution_weight.size(); i++) {
            const Classification trainingResult = current_weak_hypothesis->classify(trainingData[i].example);

            distribution_weight[i] *= std::exp(-alpha * trainingData[i].label * trainingResult);
            normalizationFactor += distribution_weight[i];
        }

        for (std::vector<weight_type>::size_type i = 0; i < distribution_weight.size(); i++) {
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
            const std::vector < LabeledExample > &training_set,
            StrongHypothesis &strong_hypothesis,
            const std::vector < WeakHypothesis * > hypothesis,
            const unsigned int maximum_iterations) {

        std::vector<weight_type> weight_distribution(training_set.size()); //holds all weight elements
        init_distribution(weight_distribution);

        do {//Main Adaboost loop

            //train weak learner and get weak hypothesis so that it minimalizes the weighted error
            weight_type weighted_error = 0;
            WeakHypothesis const * const weak_hypothesis =
                    weak_learner->learn(training_set, weight_distribution, hypothesis, weighted_error);

            //set alpha(t)
            const weight_type alpha = (weight_type)std::log( (1.0f - weighted_error)/weighted_error ) / 2.0f;

            //update the distribution
            update_distribution(alpha, weak_hypothesis, training_set, weight_distribution);

            //update the final hypothesis
            strong_hypothesis.insert(alpha, weak_hypothesis);

            t++; //next training iteration
        } while (t < maximum_iterations);
    }

};

#endif /* ADABOOST_H_ */

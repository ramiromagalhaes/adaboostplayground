#ifndef ADABOOST_H_
#define ADABOOST_H_

#include <vector>
#include <cmath>
#include <algorithm>
#include "Common.h"
#include "StrongHypothesis.h"
#include "dataprovider.h"

template<typename WeakHypothesisType>
class Adaboost {
    //TODO implement means to allow different sampling strategies
    //TODO Store errors and historic data gathered through the iterations
    //TODO devise means to implement some flexible stop criteria

protected:
    unsigned int t; /** iteration (epoch) counter */



public:
    Adaboost() : t(0) {}

    virtual ~Adaboost() {
    }

    /**
     * @brief Same as above, but now you're handling too many negative samples.
     */
    void train(
            DataProvider & training_set,
            StrongHypothesis <WeakHypothesisType> & strong_hypothesis,
            const std::vector <WeakHypothesisType> & hypothesis,
            const unsigned int maximum_iterations) {

        t = 0;

        //Vector weight_distribution holds the weights of each data sample.
        //NOTE: in this method we initialize it as proposed by Viola and Jones. The resulting vector is already normalized.
        //NOTE: the std::fill method bellow is also part of this initialization.
        WeightVector weight_distribution(training_set.size(),
                                         0.5f / training_set.sizeNegatives());
        std::fill(weight_distribution.begin(),
                  weight_distribution.begin() + training_set.sizePositives(),
                  0.5f / training_set.sizePositives());

        //this holds the weighted error of each weak classifier
        WeightVector hypothesis_weighted_errors(hypothesis.size());

        do {//Main Adaboost loop
            //train weak learner and get weak hypothesis so that it minimalizes the weighted error

            //but since we have too much data, we'll have to do this in chunks.
            //Here, the chunks are of data (positive and negative samples), since they are costly to
            //load and have it in memory. All weak hypothesis will be evaluated against a full chunk
            //of data, then another chunk will be loaded. Meanwhile, we'll have to know what are each
            //sample's weighted classification errors.

            //TODO here we could plug a way to do boosting by resampling.
            //TODO For example: we could here produce the vector we'll effectively use do the training.
            //TODO This means that the hypothesis_weighted_error vector might need to be resized

            std::fill(hypothesis_weighted_errors.begin(),
                      hypothesis_weighted_errors.end(),
                      .0f); //clean it prior to calculating the weighted errors

            const unsigned long totalIterations = training_set.size() * hypothesis.size();
            unsigned long count = 0;
            unsigned long progress = -1;

            training_set.reset();

            while ( training_set.loadNext() )
            {
                LEContainer const * const samples = training_set.getCurrentBuffer();

                for (typename std::vector <WeakHypothesisType>::size_type j = 0; j < hypothesis.size(); ++j)
                {
                    for(LEContainer::const_iterator it = samples->begin(); it != samples->end(); ++it)
                    {
                        if( hypothesis[j].classify(it->example) != it->label )
                        {
                            hypothesis_weighted_errors[j] += weight_distribution[it - samples->begin()];
                        }

                        const unsigned long currentProgress = 100 * (double)count / (double)totalIterations;
                        if (currentProgress != progress)
                        {
                            progress = currentProgress;
                            std::cout << "Adaboost iteration " << t << " in " << progress << "%.\r";
                            std::flush(std::cout);
                        }
                        ++count;
                    }
                }
            }

            //Now we must choose the weak hypothesis that produces the smallest weighted error
            //this is the final weighted_error we'll get from the best weak hypothesis found in this iteration
            const WeightVector::iterator lowest_weighted_error =
                std::min_element(hypothesis_weighted_errors.begin(), hypothesis_weighted_errors.end());
            const weight_type weighted_error = *lowest_weighted_error;

            //set alpha(t)
            weight_type alpha = (weight_type)std::log( (1.0f - weighted_error)/weighted_error ) / 2.0f;

            std::cout << "\nA new weak classifier was chosen. Will update the distribution." << std::endl;
            std::cout << "Weak classifier idx : " << lowest_weighted_error - hypothesis_weighted_errors.begin() << std::endl;
            std::cout << "Best weighted error : " << weighted_error;
            if (weighted_error > 0.5f)
            {
                std::cout << " (violates weak learning assumption)";
            }
            std::cout << "\nAlpha value         : " << alpha << std::endl;

            //Get a reference to the best weak hypothesis
            const WeakHypothesisType weak_hypothesis =
                    hypothesis[lowest_weighted_error - hypothesis_weighted_errors.begin()];

            //update the distribution
            //Since we're unable to hold the results for the selected weak hypothesis, we need to iterate
            //over the whole dataset again to update the distribution weights
            {
                training_set.reset();

                unsigned int count_correct_classification = 0;

                weight_type normalizationFactor = .0f;

                WeightVector::size_type i = 0;
                while ( training_set.loadNext() )
                {
                    LEContainer const * const samples = training_set.getCurrentBuffer();
                    for(LEContainer::const_iterator it = samples->begin(); it != samples->end(); ++it, ++i)
                    {
                        const Classification c = weak_hypothesis.classify(it->example);

                        if (c == it->label)
                        {
                            ++count_correct_classification;
                        }

                        weight_distribution[i] *= std::exp(-alpha * it->label * c);
                        normalizationFactor += weight_distribution[i];
                    }
                }

                /*
                std::transform(weight_distribution.begin(), weight_distribution.end(),
                               weight_distribution.begin(),
                               std::bind1st(std::divides<weight_type>(), normalizationFactor));
                */

                for (WeightVector::size_type i = 0; i < weight_distribution.size(); i++)
                {
                    weight_distribution[i] /= normalizationFactor;
                }

                std::cout << "Normalization factor: " << normalizationFactor << std::endl;
                std::cout << "Detection rate      : " << 100 * (double) count_correct_classification / training_set.size() << std::endl;
            }

            //update the final hypothesis
            strong_hypothesis.insert(alpha, weak_hypothesis);

            t++; //next training iteration
        } while (t < maximum_iterations);
    }

};

#endif /* ADABOOST_H_ */

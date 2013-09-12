#ifndef ADABOOST_H_
#define ADABOOST_H_

#include <vector>
#include <cmath>
#include <algorithm>

#include "common.h"
#include "stronghypothesis.h"
#include "dataprovider.h"

template<typename WeakHypothesisType>
class Adaboost {
    //TODO implement means to allow different sampling strategies
    //TODO Store errors and historic data gathered through the iterations
    //TODO devise means to implement some flexible stop criteria

protected:
    /** iteration (epoch) counter */
    unsigned int t;



public:
    Adaboost() : t(0) {}

    virtual ~Adaboost() {
    }

    /**
     * @brief This method trains a strong classifier.
     * @param training_set A vector of LabeledExamples that will be used in training.
     * @param strong_hypothesis The object that will hold the strong classifier.
     * @param maximum_iterations The maximum iterations that this training will perform.
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

            { //in this block we pick the best weak classifier
                training_set.reset();
                LabeledExample sample; //TODO should probably move this integralSum calculation to the LabeledExample class
                for(WeightVector::size_type i = 0; training_set.nextSample(sample); ++i ) //i refers to the samples
                {
                    for (typename std::vector <WeakHypothesisType>::size_type j = 0; j < hypothesis.size(); ++j) //j refers to the classifiers
                    {
                        //Might be faster than an if thanks to branch prediction
                        //See: http://stackoverflow.com/questions/11227809/why-is-processing-a-sorted-array-faster-than-an-unsorted-array
                        hypothesis_weighted_errors[j] += weight_distribution[i]
                                * (hypothesis[j].classify(sample) != sample.label);

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

                LabeledExample sample;
                for( WeightVector::size_type i = 0; training_set.nextSample(sample); ++i ) //i refers to the samples
                {
                    const Classification c = weak_hypothesis.classify(sample);
                    count_correct_classification += (c == sample.label); //increment if correctly classified

                    weight_distribution[i] *= std::exp(-alpha * sample.label * c);
                    normalizationFactor += weight_distribution[i];
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
                std::cout << "Detection rate      : " << 100 * (double) count_correct_classification / training_set.size() << '%' << std::endl;
            }

            //update the final hypothesis
            strong_hypothesis.insert(alpha, weak_hypothesis);

            t++; //next training iteration
        } while (t < maximum_iterations);
    }

};

#endif /* ADABOOST_H_ */

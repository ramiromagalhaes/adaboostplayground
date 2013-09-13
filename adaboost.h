#ifndef ADABOOST_H_
#define ADABOOST_H_

#include <vector>
#include <cmath>
#include <algorithm>

#include "common.h"
#include "stronghypothesis.h"
#include "dataprovider.h"



/**
 * A callback to report the progress of the Adaboost train method. Just create
 * your implementation and pass an instance of it to the Adaboost constructor.
 */
struct ProgressCallback
{
    virtual void tick (const unsigned int iteration,
                       const unsigned long current,
                       const unsigned long total) =0;

    virtual void classifierSelected (const weight_type alpha,
                                     const weight_type normalization_factor,
                                     const weight_type lowest_classifier_error,
                                     const unsigned int classifier_idx) =0;
};



template<typename WeakHypothesisType>
class Adaboost {
    //TODO implement means to allow different sampling strategies
    //TODO Store errors and historic data gathered through the iterations
    //TODO devise means to implement some flexible stop criteria

protected:
    /** iteration (epoch) counter */
    unsigned int t;

    ProgressCallback * progressCallback;


public:
    Adaboost() : t(0), progressCallback(0) {}

    Adaboost(ProgressCallback * progressCallback_) : t(0),
                                                     progressCallback(progressCallback_) {}

    virtual ~Adaboost() {}

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
        WeightVector weight_distribution(training_set.size());
            std::fill(weight_distribution.begin(),
                  weight_distribution.begin() + training_set.sizePositives(),
                  0.5f / training_set.sizePositives());
            std::fill(weight_distribution.begin() + training_set.sizePositives(),
                  weight_distribution.end(),
                  0.5f / training_set.sizeNegatives());

        //This holds the weighted error of each weak classifier.
        WeightVector hypothesis_weighted_errors(hypothesis.size());

        do {//Main Adaboost loop
            unsigned long count = 0; //this and totalIterations help track the progress of the weak learner
            const unsigned long totalIterations = training_set.size() * hypothesis.size();

            //Train weak learner and get weak hypothesis so that it "minimalizes" the weighted error.

            //TODO here we could plug a way to do boosting by resampling. For example: we could here produce the vector we'll effectively use do the training. This means that the hypothesis_weighted_error vector might need to be resized

            std::fill(hypothesis_weighted_errors.begin(),
                      hypothesis_weighted_errors.end(), 0); //clean this prior to calculating the weighted errors

            {//in this block we pick the best weak classifier
                training_set.reset();
                LabeledExample sample;
                for(WeightVector::size_type i = 0; training_set.nextSample(sample); ++i ) //i refers to the samples
                {
                    for (typename std::vector <WeakHypothesisType>::size_type j = 0; j < hypothesis.size(); ++j) //j refers to the classifiers
                    {
                        //Might be faster than an if thanks to branch prediction
                        //See: http://stackoverflow.com/questions/11227809/why-is-processing-a-sorted-array-faster-than-an-unsorted-array
                        hypothesis_weighted_errors[j] += weight_distribution[i]
                                * (hypothesis[j].classify(sample) != sample.label);

                        if (progressCallback)
                        {
                            ++count;
                            progressCallback->tick(t, count, totalIterations);
                        }
                    }
                }
            }



            //Now we must choose the weak hypothesis that produces the smallest weighted error this is
            //the final weighted_error we'll get from the best weak hypothesis found in this iteration.
            const WeightVector::iterator lowest_weighted_error =
                    std::min_element(hypothesis_weighted_errors.begin(),
                                     hypothesis_weighted_errors.end());
            const weight_type weighted_error = *lowest_weighted_error;

            //Get a reference to the best weak hypothesis
            const WeakHypothesisType weak_hypothesis =
                    hypothesis[lowest_weighted_error - hypothesis_weighted_errors.begin()];

            //set alpha(t)
            weight_type alpha = (weight_type)std::log( (1.0f - weighted_error)/weighted_error ) / 2.0f;



            //Now we just have to update the weight distribution of the samples.
            //Normalization factor is not inside the block because we report it to the progressCallback.
            weight_type normalizationFactor = 0;
            {
                training_set.reset();
                LabeledExample sample;
                for( WeightVector::size_type i = 0; training_set.nextSample(sample); ++i ) //i refers to the samples
                {
                    weight_distribution[i] *= std::exp(-alpha * sample.label * weak_hypothesis.classify(sample));
                    normalizationFactor += weight_distribution[i];
                }

                std::transform(weight_distribution.begin(), weight_distribution.end(),
                               weight_distribution.begin(),
                               std::bind2nd(std::divides<weight_type>(), normalizationFactor)); // bind2nd makes normalizationFactor the divisor. see also bind1st.
            }



            if (progressCallback)
            {
                progressCallback->classifierSelected(alpha,
                                                     normalizationFactor,
                                                     weighted_error,
                                                     lowest_weighted_error - hypothesis_weighted_errors.begin());
            }



            //update the final hypothesis
            strong_hypothesis.insert(alpha, weak_hypothesis);

            t++; //next training iteration
        } while (t < maximum_iterations);
    }

};

#endif /* ADABOOST_H_ */

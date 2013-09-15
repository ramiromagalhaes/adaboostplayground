#ifndef ADABOOST_H_
#define ADABOOST_H_

#include <vector>
#include <cmath>
#include <algorithm>

#include "labeledexample.h"
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



/**
 * A simple implementation of a ProgressCallback.
 */
class SimpleProgressCallback : public ProgressCallback
{
private:
    int progress;

public:
    SimpleProgressCallback() : progress(-1) {}

    virtual void tick (const unsigned int iteration, const unsigned long current, const unsigned long total)
    {
        const int currentProgress = (int) (100 * current / total);
        if (currentProgress != progress)
        {
            progress = currentProgress;
            std::cout << "Adaboost iteration " << iteration << " in " << progress << "%.\r";
            std::flush(std::cout);
        }
    }

    virtual void classifierSelected (const weight_type alpha,
                                     const weight_type normalization_factor,
                                     const weight_type lowest_classifier_error,
                                     const unsigned int classifier_idx)
    {
        std::cout << "\nA new weak classifier was chosen." << std::endl;
        std::cout <<   "  Weak classifier idx : " << classifier_idx << std::endl;
        std::cout <<   "  Best weighted error : " << lowest_classifier_error;
        if (lowest_classifier_error > 0.5f)
        {
            std::cout << " (violates weak learning assumption)";
        }
        std::cout << "\n  Alpha value         : " << alpha << std::endl;
        std::cout <<   "  Normalization factor: " << normalization_factor << std::endl;
    }
};



template<typename WeakHypothesisType>
class Adaboost {
    //TODO implement means to allow different weak learner boosting strategies: reweighting and resampling
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

    void train(
            std::vector<LabeledExample> positiveSamples,
            std::vector<LabeledExample> negativeSamples,
            StrongHypothesis <WeakHypothesisType> & strong_hypothesis,
            const std::vector <WeakHypothesisType> & hypothesis,
            const unsigned int maximum_iterations)
    {
        t = 0;


        const unsigned int total_samples = positiveSamples.size() + negativeSamples.size();
        std::vector<LabeledExample *> allSamples(total_samples); //TODO make this pointer constant might be a good idea
        {//I'm pretty sure there is a better way to do this. I even tried std::transform with a functor of my own. Did not work, so back to basics.
            unsigned int i = 0;
            for (; i < positiveSamples.size(); ++i)
            {
                allSamples[i] = &(positiveSamples[i]);
            }
            for (; i - positiveSamples.size() < negativeSamples.size(); ++i)
            {
                allSamples[i] = &(negativeSamples[i - positiveSamples.size()]);
            }
        }
        for (int i = 0; i < allSamples.size(); ++i)
        {
            if ( !allSamples[i] )
            {
                std::cout << i << ' ';
            }
        }


        //Vector weight_distribution holds the weights of each data sample.
        WeightVector weight_distribution(total_samples);
        std::fill(weight_distribution.begin(),  weight_distribution.begin() + positiveSamples.size(),
              0.5f / positiveSamples.size());
        std::fill(weight_distribution.begin() + positiveSamples.size(), weight_distribution.end(),
              0.5f / negativeSamples.size());


        //This holds the weighted error of each weak classifier.
        WeightVector hypothesis_weighted_errors(hypothesis.size());


        do {//Main Adaboost loop

            {//Train weak learner and get weak hypothesis so that it "minimalizes" the weighted error.
                unsigned long count = 0; //Counts how many images have already been iterated over. Will be used by the progressCallback.
                if (progressCallback) //Initial report of the progress.
                {
                    ++count;
                    progressCallback->tick(t, count, total_samples);
                }

                std::fill(hypothesis_weighted_errors.begin(),
                          hypothesis_weighted_errors.end(), 0); //clean this prior to calculating the weighted errors

                //In this block we calculate the weighted errors of each weak classifier with respect to the weights of each instance
                for(WeightVector::size_type i = 0; i < total_samples; ++i ) //i refers to the samples
                {
                    LabeledExample * const sample = allSamples[i];
                    for (typename std::vector <WeakHypothesisType>::size_type j = 0; j < hypothesis.size(); ++j) //j refers to the classifiers
                    {
                        hypothesis_weighted_errors[j] += weight_distribution[i]
                                * (hypothesis[j].classify(*sample) != sample->label);
                    }

                    if (progressCallback)
                    {
                        ++count;
                        progressCallback->tick(t, count, total_samples);
                    }
                }
            }



            //Now we choose the weak hypothesis with the smallest weighted.
            const WeightVector::iterator lowest_weighted_error =
                    std::min_element(hypothesis_weighted_errors.begin(),
                                     hypothesis_weighted_errors.end());
            const weight_type weighted_error = *lowest_weighted_error;

            //Get a reference to the best weak hypothesis
            const WeakHypothesisType weak_hypothesis =
                    hypothesis[lowest_weighted_error - hypothesis_weighted_errors.begin()];

            //Set alpha for this iteration
            weight_type alpha = (weight_type)std::log( (1.0f - weighted_error)/weighted_error ) / 2.0f;



            //Now we just have to update the weight distribution of the samples.
            //Normalization factor is not inside the block because we report it to the progressCallback.
            weight_type normalizationFactor = 0;
            {
                for( WeightVector::size_type i = 0; i < total_samples; ++i ) //i refers to the weight of the samples
                {
                    LabeledExample * const sample = allSamples[i];

                    weight_distribution[i] *= std::exp(-alpha * sample->label * weak_hypothesis.classify(*sample));
                    normalizationFactor += weight_distribution[i];
                }

                std::transform(weight_distribution.begin(), weight_distribution.end(),
                               weight_distribution.begin(),
                               std::bind2nd(std::divides<weight_type>(), normalizationFactor)); //bind2nd makes normalizationFactor the divisor.
                                                                                                //see also bind1st.
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



    /**
     * @brief This method trains a strong classifier.
     * @param training_set A vector of LabeledExamples that will be used in training.
     * @param strong_hypothesis The object that will hold the strong classifier.
     * @param maximum_iterations The maximum iterations that this training will perform.
     *
     * Initializes the weight distribution of the training set using Viola and Jones method,
     * instead of the original one proposed by Freund and Schapire. The weak learner is boosted
     * by reweighting, instead of resampling.
     */
    void train(
            DataProvider & training_set,
            StrongHypothesis <WeakHypothesisType> & strong_hypothesis,
            const std::vector <WeakHypothesisType> & hypothesis,
            const unsigned int maximum_iterations)
    {
        t = 0;

        //Vector weight_distribution holds the weights of each data sample.
        WeightVector weight_distribution(training_set.size());
            std::fill(weight_distribution.begin(),  weight_distribution.begin() + training_set.sizePositives(),
                  0.5f / training_set.sizePositives());
            std::fill(weight_distribution.begin() + training_set.sizePositives(), weight_distribution.end(),
                  0.5f / training_set.sizeNegatives());

        //This holds the weighted error of each weak classifier.
        WeightVector hypothesis_weighted_errors(hypothesis.size());

        do {//Main Adaboost loop

            unsigned long count = 0; //Counts how many images have already been iterated over. Will be used by the progressCallback.
            if (progressCallback) //Initial report of the progress.
            {
                ++count;
                progressCallback->tick(t, count, training_set.size());
            }


            //Train weak learner and get weak hypothesis so that it "minimalizes" the weighted error.
            std::fill(hypothesis_weighted_errors.begin(),
                      hypothesis_weighted_errors.end(), 0); //clean this prior to calculating the weighted errors

            {//In this block we calculate the weighted errors of each weak classifier with respect to the weights of each instance
                training_set.reset();
                LabeledExample sample;
                for(WeightVector::size_type i = 0; training_set.nextSample(sample); ++i ) //i refers to the samples
                {
                    for (typename std::vector <WeakHypothesisType>::size_type j = 0; j < hypothesis.size(); ++j) //j refers to the classifiers
                    {
                        hypothesis_weighted_errors[j] += weight_distribution[i]
                                * (hypothesis[j].classify(sample) != sample.label);
                    }

                    if (progressCallback)
                    {
                        ++count;
                        progressCallback->tick(t, count, training_set.size());
                    }
                }
            }



            //Now we choose the weak hypothesis with the smallest weighted.
            const WeightVector::iterator lowest_weighted_error =
                    std::min_element(hypothesis_weighted_errors.begin(),
                                     hypothesis_weighted_errors.end());
            const weight_type weighted_error = *lowest_weighted_error;

            //Get a reference to the best weak hypothesis
            const WeakHypothesisType weak_hypothesis =
                    hypothesis[lowest_weighted_error - hypothesis_weighted_errors.begin()];

            //Set alpha for this iteration
            weight_type alpha = (weight_type)std::log( (1.0f - weighted_error)/weighted_error ) / 2.0f;



            //Now we just have to update the weight distribution of the samples.
            //Normalization factor is not inside the block because we report it to the progressCallback.
            weight_type normalizationFactor = 0;
            {
                training_set.reset();
                LabeledExample sample;
                for( WeightVector::size_type i = 0; training_set.nextSample(sample); ++i ) //i refers to the weight of the samples
                {
                    weight_distribution[i] *= std::exp(-alpha * sample.label * weak_hypothesis.classify(sample));
                    normalizationFactor += weight_distribution[i];
                }

                std::transform(weight_distribution.begin(), weight_distribution.end(),
                               weight_distribution.begin(),
                               std::bind2nd(std::divides<weight_type>(), normalizationFactor)); //bind2nd makes normalizationFactor the divisor.
                                                                                                //see also bind1st.
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

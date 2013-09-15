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
    virtual void beginAdaboostIteration(const unsigned int iteration) =0;

    virtual void tick (const unsigned long current, const unsigned long total) =0;

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

    virtual void beginAdaboostIteration(const unsigned int iteration)
    {
        std::cout << "Adaboost iteration " << iteration << ".\n";
    }

    virtual void tick (const unsigned long current, const unsigned long total)
    {
        const int currentProgress = (int) (100 * current / total);
        if (currentProgress != progress)
        {
            progress = currentProgress;
            std::cout << "Progress: " << progress << "%.\r";
            std::flush(std::cout);
        }
    }

    virtual void classifierSelected (const weight_type alpha,
                                     const weight_type normalization_factor,
                                     const weight_type lowest_classifier_error,
                                     const unsigned int classifier_idx)
    {
        std::cout << "\rA new a weak classifier was chosen." << std::endl;
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



/**
 * Implementation of the Adaboost algorithm.
 */
template<typename WeakHypothesisType>
class Adaboost {
    //TODO implement means to allow different weak learner boosting strategies: reweighting and resampling
    //TODO Store errors and historic data gathered through the iterations
    //TODO devise means to implement some flexible stop criteria

protected:
    /** iteration (epoch) counter */
    unsigned int t;

    /** Its methods will be invoked to report the algorithm progress */
    ProgressCallback * progressCallback;



    /**
     * Returns the normalization factor so it can be displayed to the user.
     */
    weight_type updateWeightDistribution( const std::vector<LabeledExample *> & allSamples,
                                          const weight_type alpha,
                                          const WeakHypothesisType & selected_hypothesis,
                                          WeightVector & weight_distribution )
    {
        weight_type normalizationFactor = 0;

        for( WeightVector::size_type i = 0; i < allSamples.size(); ++i ) //i refers to the weight of the samples
        {
            LabeledExample * const sample = allSamples[i];

            weight_distribution[i] *= std::exp(-alpha * sample->label * selected_hypothesis.classify(*sample));
            normalizationFactor += weight_distribution[i];
        }

        std::transform(weight_distribution.begin(), weight_distribution.end(),
                       weight_distribution.begin(),
                       std::bind2nd(std::divides<weight_type>(), normalizationFactor)); //bind2nd makes normalizationFactor the divisor.
                                                                                        //see also bind1st.

        return normalizationFactor;
    }



    /**
     * Used to produce a pointer from an object.
     */
    struct ToPointer
    {
        inline LabeledExample * operator()(LabeledExample & ex)
        {
            return &ex;
        }
    };



    /**
     * The weak learner used in this Adaboost implementation.
     */
    struct WeakLearner
    {
        //This holds the weighted error of each weak classifier.
        WeightVector hypothesis_weighted_errors;

        void operator()( const std::vector<LabeledExample *> & allSamples,
                         const std::vector<WeakHypothesisType> & hypothesis,
                         const WeightVector & weight_distribution,
                         weight_type & selected_weak_hypothesis_weighted_error,
                         unsigned int & selected_weak_hypothesis_index,
                         ProgressCallback * const progressCallback = 0)
        {
            unsigned long count = 0; //Counts how many images have already been iterated over. Will be used by the progressCallback.

            hypothesis_weighted_errors.resize(hypothesis.size());//It should be effectivelly resized only once
            std::fill(hypothesis_weighted_errors.begin(),
                      hypothesis_weighted_errors.end(), 0); //But should be cleaned at all times.

            //Now we calculate the weighted errors of each weak classifier with respect to the weights of each instance
            for (typename std::vector <WeakHypothesisType>::size_type j = 0; j < hypothesis.size(); ++j) //j refers to the classifiers
            {
                for(WeightVector::size_type i = 0; i < allSamples.size(); ++i ) //i refers to the samples
                {
                    hypothesis_weighted_errors[j] += weight_distribution[i]
                            * (hypothesis[j].classify(*(allSamples[i])) != allSamples[i]->label);
                }

                if (progressCallback)
                {
                    ++count;
                    progressCallback->tick(count, allSamples.size());
                }
            }

            //Now we choose the weak hypothesis with the smallest weighted.
            const WeightVector::iterator lowest_weighted_error =
                    std::min_element(hypothesis_weighted_errors.begin(),
                                     hypothesis_weighted_errors.end());

            //This will hold the weighted error of the best weak classifier found by the weak learner
            selected_weak_hypothesis_weighted_error = *lowest_weighted_error;

            //This will hold a reference to the best weak hypothesis found by the weak learner
            selected_weak_hypothesis_index = lowest_weighted_error - hypothesis_weighted_errors.begin();
        }
    };



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


        //Collects into allSamples pointers to both positive and negative LabeledExamples
        std::vector<LabeledExample *> allSamples(positiveSamples.size() + negativeSamples.size());//TODO make the LabeledExample pointer constant?
        std::transform(positiveSamples.begin(), positiveSamples.end(),
                       allSamples.begin(),
                       ToPointer());
        std::transform(negativeSamples.begin(), negativeSamples.end(),
                       allSamples.begin() + positiveSamples.size(),
                       ToPointer());


        //Vector weight_distribution holds the weights of each data sample.
        WeightVector weight_distribution(allSamples.size());
        std::fill(weight_distribution.begin(),  weight_distribution.begin() + positiveSamples.size(),
                  0.5f / positiveSamples.size());
        std::fill(weight_distribution.begin() + positiveSamples.size(), weight_distribution.end(),
                  0.5f / negativeSamples.size());


        //Initialize the weak learner.
        //TODO Provide the WeakLearner to Adaboost
        Adaboost::WeakLearner weakLearner;


        do {//Main Adaboost loop
            if(progressCallback)
            {
                progressCallback->beginAdaboostIteration(t);
            }

            //This holds the weighted error of the best weak classifier. The weak lerner sets it.
            weight_type weighted_error = 0;

            //This holds the index of the best weak hypothesis. The weak lerner sets it.
            unsigned int weak_hypothesis_index = 0;

            //Train weak learner and get weak hypothesis so that it "minimalizes" the weighted error.
            weakLearner(allSamples,
                        hypothesis,
                        weight_distribution,
                        weighted_error,
                        weak_hypothesis_index,
                        progressCallback);


            //Set alpha for this iteration
            const weight_type alpha = (weight_type)std::log( (1.0f - weighted_error)/weighted_error ) / 2.0f;


            //Now we just have to update the weight distribution of the samples.
            //Normalization factor is not inside the block because we report it to the progressCallback.
            weight_type normalizationFactor =
                    updateWeightDistribution( allSamples,
                                              alpha,
                                              hypothesis[weak_hypothesis_index],
                                              weight_distribution );

            if (progressCallback)
            {
                progressCallback->classifierSelected(alpha,
                                                     normalizationFactor,
                                                     weighted_error,
                                                     weak_hypothesis_index);
            }


            //update the final hypothesis
            strong_hypothesis.insert(alpha, hypothesis[weak_hypothesis_index]);

            t++; //next training iteration
        } while (t < maximum_iterations);
    }
};

#endif /* ADABOOST_H_ */

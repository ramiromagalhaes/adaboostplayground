#ifndef ADABOOST_H_
#define ADABOOST_H_

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <tbb/tbb.h>

#include "common.h"
#include "labeledexample.h"
#include "stronghypothesis.h"
#include "progresscallback.h"

#include "decisionstumpweaklearner.h"



/**
 * Implementation of the Adaboost algorithm.
 */
template<typename WeakHypothesisType>
class Adaboost {
    //TODO implement means to allow different weak learner boosting strategies: reweighting and resampling
    //TODO Store errors and historic data gathered through the iterations
    //TODO devise means to implement some flexible stop criteria

protected:
    /** Its methods will be invoked to report the algorithm progress */
    ProgressCallback * progressCallback;



    /**
     * Defines the mutex type the Weak Learner will use.
     */
    WeakLearnerMutex weak_learner_mutex;



    /**
     * Returns the normalization factor so it can be displayed to the user.
     */
    weight_type updateWeightDistribution( const std::vector<const LabeledExample *> & allSamples,
                                          const weight_type alpha,
                                          const WeakHypothesisType & selected_hypothesis,
                                          WeightVector & weight_distribution )
    {
        weight_type normalizationFactor = 0;

        for( WeightVector::size_type i = 0; i < allSamples.size(); ++i )
        {
            Classification c = selected_hypothesis.classify( *(allSamples[i]) );

            //This is the original Adaboost weight update. Viola and Jones report a slightly different equation,
            //but their starting weights are a little different too.
            weight_distribution[i] = weight_distribution[i] * std::exp(-alpha * (allSamples[i]->getLabel()) * c);
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
        inline LabeledExample * operator()(LabeledExample & ex) const
        {
            return &ex;
        }
    };



public:
    Adaboost() : progressCallback(new SimpleProgressCallback()),
                 weak_learner_mutex() {}

    Adaboost(ProgressCallback * progressCallback_) : progressCallback(progressCallback_),
                                                     weak_learner_mutex() {}

    ~Adaboost() {
        if ( !progressCallback )
        {
            delete progressCallback;
        }
    }

    /**
     *
     * @return true if reached maximum_iterations when returning, of false otherwise.
     */
    bool train(std::vector<LabeledExample> positiveSamples,
               std::vector<LabeledExample> negativeSamples,
               StrongHypothesis<WeakHypothesisType> & strong_hypothesis,
               std::vector <WeakHypothesisType> & hypothesis,
               const unsigned int maximum_iterations)
    {
        unsigned int t = 0;


        //allSamples collects pointers to both positive and negative LabeledExamples
        std::vector<const LabeledExample *> allSamples(positiveSamples.size() + negativeSamples.size());//TODO make the LabeledExample pointer constant?
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


        do {//Main Adaboost loop
            if(progressCallback)
            {
                progressCallback->beginAdaboostIteration(t);
            }

            //Holds the weighted error of the best weak classifier selected this boosting round. The weak lerner sets it.
            weight_type weighted_error = std::numeric_limits<weight_type>::max();

            //Holds the index of the best weak hypothesis. The weak lerner sets it.
            unsigned int weak_hypothesis_index = 0;

            //A progress counter
            unsigned long count = 0;

            //Train weak learner and get weak hypothesis so that it "minimalizes" the weighted error.
            tbb::parallel_for( tbb::blocked_range< unsigned int >(0, hypothesis.size()),
                               DecisionStumpWeakLearner<WeakHypothesisType>(&weak_learner_mutex,
                                                                            &allSamples,
                                                                            &weight_distribution,
                                                                            &hypothesis,
                                                                            &weighted_error,
                                                                            &weak_hypothesis_index,
                                                                            &count,
                                                                            progressCallback) );

            //Set alpha for this iteration
            const weight_type alpha = 0.5f * std::log( (1.0f - weighted_error) / weighted_error );
            if ( std::isnan(alpha) || std::isinf(alpha) )
            {
                std::cout << "Exiting trainning loop since alpha is infinity or not a number." << std::endl;
                return false;
            }

            //Now we just have to update the weight distribution of the samples.
            //Normalization factor is not inside the block because we report it to the progressCallback.
            const weight_type normalizationFactor =
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

        return true;
    }
};

#endif /* ADABOOST_H_ */

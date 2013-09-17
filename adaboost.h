#ifndef ADABOOST_H_
#define ADABOOST_H_

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <tbb/tbb.h>

#include "labeledexample.h"
#include "common.h"
#include "stronghypothesis.h"



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
        std::cout << "Adaboost iteration " << iteration << std::endl;
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
        std::cout << "\rA new a weak classifier was chosen.";
        std::cout << "\n  Weak classifier idx : " << classifier_idx;
        std::cout << "\n  Best weighted error : " << lowest_classifier_error;
        if (lowest_classifier_error > 0.5f)
        {
            std::cout << " (violates weak learning assumption)";
        }
        std::cout << "\n  Alpha value         : " << alpha;
        std::cout << "\n  Normalization factor: " << normalization_factor << std::endl;
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

            /*
            if ( sample->label ==  selected_hypothesis.classify(*sample) )
            {
                weight_distribution[i] *= std::exp(-2.0f * alpha);
            }
            */
            //this is the original Adaboost weight update
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
    struct ParallelWeakLearner
    {
        struct feature_and_weight
        {
            float feature;
            weight_type weight;
            Classification label;
        };

        struct compare_feature
        {
            bool operator()(const feature_and_weight & lh, const feature_and_weight &rh) const
            {
                return lh.feature > rh.feature;
            }
        };

        const std::vector<LabeledExample *> * allSamples;
        std::vector<WeakHypothesisType> * hypothesis;
        const WeightVector * weight_distribution;
        weight_type * selected_weak_hypothesis_weighted_error;
        unsigned int * selected_weak_hypothesis_index;
        unsigned long * count;
        ProgressCallback * const progressCallback;

        ParallelWeakLearner(const std::vector<LabeledExample *> * allSamples_,
                            std::vector<WeakHypothesisType> * hypothesis_,
                            const WeightVector * weight_distribution_,
                            weight_type * selected_weak_hypothesis_weighted_error_,
                            unsigned int * selected_weak_hypothesis_index_,
                            unsigned long * count_,
                            ProgressCallback * const progressCallback_) : allSamples(allSamples_),
                                                                          hypothesis(hypothesis_),
                                                                          weight_distribution(weight_distribution_),
                                                                          selected_weak_hypothesis_weighted_error(selected_weak_hypothesis_weighted_error_),
                                                                          selected_weak_hypothesis_index(selected_weak_hypothesis_index_),
                                                                          count(count_),
                                                                          progressCallback(progressCallback_)
        {
            *selected_weak_hypothesis_weighted_error = std::numeric_limits<float>::max();
            *count = 0;
        }

        /**
         * The main loop will run over the hypothesis vector.
         */
        void operator()(tbb::blocked_range< unsigned int > & range) const
        {
            //Calculate the weighted errors of each weak classifier with respect to the weights of each instance
            for (unsigned int j = range.begin(); j < range.end(); ++j) //j refers to the classifiers
            {
                //Feature values and respective weight and label
                std::vector<feature_and_weight> feature_values(allSamples->size());

                //========= BEGIN WTF ZONE =========
                //For a nice explanation about what is going on bellow, refer to Schapire and Freund's Boosting book, section 3.4.2
                weight_type total_w_1_p = 0;
                weight_type total_w_1_n = 0;
                for(WeightVector::size_type i = 0; i < feature_values.size(); ++i ) //i refers to the samples
                {
                    feature_values[i].feature = (*hypothesis)[j].featureValue( *( (*allSamples)[i] ) );
                    feature_values[i].label = (*allSamples)[i]->label;
                    feature_values[i].weight = (*weight_distribution)[i];

                    total_w_1_p += (*weight_distribution)[i] * ((*allSamples)[i]->label == yes);
                    total_w_1_n += (*weight_distribution)[i] * ((*allSamples)[i]->label == no);
                }

                std::sort( feature_values.begin(), feature_values.end(), compare_feature() );

                weight_type total_w_0_p = 0;
                weight_type total_w_0_n = 0;

                float best_error = std::min(total_w_1_n, total_w_1_p);

                float v = feature_values[0].feature;
                Classification c0 = total_w_0_n <= total_w_0_p ? yes : no;
                Classification c1 = total_w_1_n <= total_w_1_p ? yes : no;

                for(WeightVector::size_type k = 0; k < feature_values.size(); ++k )
                {
                    total_w_0_p += feature_values[k].weight * (feature_values[k].label == yes);
                    total_w_0_n += feature_values[k].weight * (feature_values[k].label == no);

                    total_w_1_p -= feature_values[k].weight * (feature_values[k].label == yes);
                    total_w_1_n -= feature_values[k].weight * (feature_values[k].label == no);

                    const float error = std::min(total_w_0_n, total_w_0_p) + std::min(total_w_1_n, total_w_1_p);

                    if (error < best_error)
                    {
                        best_error = error;

                        v = feature_values[k].feature;

                        c0 = total_w_0_n <= total_w_0_p ? yes : no;
                        c1 = total_w_1_n <= total_w_1_p ? yes : no;
                    }
                }
                //Ok... That was what's in the book. Give me back the controls now.
                //========= END WTF ZONE =========

                (*hypothesis)[j].setThreshold(v); //If we had a hypothesis[j].setP() (Viola and Jones's polarity), we could provide set it with c0.

                {
                    //this must be synchonized way
                    tbb::queuing_mutex selected_weak_hypothesis_mutex; //TODO maybe a spin_lock is better here?
                    tbb::queuing_mutex::scoped_lock lock(selected_weak_hypothesis_mutex);
                    if (best_error < *selected_weak_hypothesis_weighted_error)
                    {
                        *selected_weak_hypothesis_weighted_error = best_error;
                        *selected_weak_hypothesis_index = j;
                    }
                    lock.release();
                }

                //synchronization should happen INSIDE the if
                if (progressCallback)
                {
                    tbb::queuing_mutex progress_mutex;//TODO maybe a spin_lock is better here?
                    tbb::queuing_mutex::scoped_lock lock(progress_mutex);
                    ++(*count);
                    progressCallback->tick(*count, hypothesis->size());
                    lock.release();
                }
            }
        }
    };



public:
    Adaboost() : progressCallback(0) {}

    Adaboost(ProgressCallback * progressCallback_) : progressCallback(progressCallback_) {}

    virtual ~Adaboost() {}

    /**
     *
     * @return true if reached maximum_iterations when returning, of false otherwise.
     */
    bool train(std::vector<LabeledExample> positiveSamples,
               std::vector<LabeledExample> negativeSamples,
               StrongHypothesis & strong_hypothesis,
               std::vector <WeakHypothesisType> & hypothesis,
               const unsigned int maximum_iterations)
    {
        unsigned int t = 0;


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


        do {//Main Adaboost loop
            if(progressCallback)
            {
                progressCallback->beginAdaboostIteration(t);
            }

            //This holds the weighted error of the best weak classifier. The weak lerner sets it.
            weight_type weighted_error = 0;

            //This holds the index of the best weak hypothesis. The weak lerner sets it.
            unsigned int weak_hypothesis_index = 0;

            //A progress counter
            unsigned long count;

            //Train weak learner and get weak hypothesis so that it "minimalizes" the weighted error.
            tbb::parallel_for( tbb::blocked_range< unsigned int >(0, hypothesis.size()),
                               Adaboost<WeakHypothesisType>::ParallelWeakLearner(&allSamples,
                                                                                 &hypothesis,
                                                                                 &weight_distribution,
                                                                                 &weighted_error,
                                                                                 &weak_hypothesis_index,
                                                                                 &count,
                                                                                 progressCallback) );

            //Set alpha for this iteration
            const weight_type alpha = (weight_type)std::log( (1.0f - weighted_error)/weighted_error ) / 2.0f;
            if ( std::isnan(alpha) || std::isinf(alpha) )
            {
                std::cout << "Exiting trainning loop since alpha is infinity or not a number." << std::endl;
                return false;
            }


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

        return true;
    }
};

#endif /* ADABOOST_H_ */

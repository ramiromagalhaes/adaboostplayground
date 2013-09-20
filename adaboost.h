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



//This is usefull for certain debugging tasks. Uncomment if needed.
//tbb::task_scheduler_init init(1);



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
        std::cout << "Adaboost iteration " << iteration << '\n';
        std::cout.flush();
    }

    virtual void tick (const unsigned long current, const unsigned long total)
    {
        const int currentProgress = (int) (100 * current / total);
        if (currentProgress != progress)
        {
            progress = currentProgress;
            std::cout << "Progress: " << progress << "%.\r";
            std::cout.flush();
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
        std::cout << "\n  Normalization factor: " << normalization_factor << '\n';
        std::cout.flush();
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
     * Defines the mutex type the Weak Learner will use.
     */
    typedef tbb::queuing_mutex WeakLearnerMutex;
    WeakLearnerMutex weak_learner_mutex;

    /**
     * Implements a weak learner which weak classifiers are decision stumps.
     */
    struct DecisionStumpWeakLearner
    {
        struct feature_and_weight
        {
            float feature;
            weight_type weight;
            Classification label;

            bool operator < (const feature_and_weight & f) const
            {
                return feature < f.feature;
            }
        };

                                 WeakLearnerMutex * const mutex;
        const std::vector<const LabeledExample *> * const allSamples;
                               const WeightVector * const weight_distribution;
                  std::vector<WeakHypothesisType> * const hypothesis;
                                      weight_type * const selected_weak_hypothesis_weighted_error;
                                     unsigned int * const selected_weak_hypothesis_index;
                                    unsigned long * const count;
                                 ProgressCallback * const progressCallback;

         DecisionStumpWeakLearner(WeakLearnerMutex * const mutex_,
         const std::vector<const LabeledExample *> * const allSamples_,
                                const WeightVector * const weight_distribution_,
                   std::vector<WeakHypothesisType> * const hypothesis_,
                                       weight_type * const selected_weak_hypothesis_weighted_error_,
                                      unsigned int * const selected_weak_hypothesis_index_,
                                     unsigned long * const count_,
                                  ProgressCallback * const progressCallback_) : mutex(mutex_),
                                                                                allSamples(allSamples_),
                                                                                weight_distribution(weight_distribution_),
                                                                                hypothesis(hypothesis_),
                                                                                selected_weak_hypothesis_weighted_error(selected_weak_hypothesis_weighted_error_),
                                                                                selected_weak_hypothesis_index(selected_weak_hypothesis_index_),
                                                                                count(count_),
                                                                                progressCallback(progressCallback_) {}

        /**
         * The main loop will run over the hypothesis vector.
         */
        void operator()(tbb::blocked_range< unsigned int > & range) const
        {
            //Feature values and respective weight and label
            std::vector<feature_and_weight> feature_values(allSamples->size());

            //Calculate the weighted errors of each weak classifier with respect to the weights of each instance
            for (unsigned int j = range.begin(); j < range.end(); ++j) //j refers to the classifiers
            {
                //========= BEGIN WTF ZONE =========
                //For an explanation about what is going on bellow, refer to Schapire and Freund's Boosting book, chapter 3.4.2
                weight_type total_w_1_p = 0; //remaining true positives for feature_values above k. This is Viola and Jones' (T+ - S+), as seen in section 3.1.
                weight_type total_w_1_n = 0; //remaining false positives for feature_values above k. This is Viola and Jones' (T- - S-).
                for(WeightVector::size_type i = 0; i < feature_values.size(); ++i ) //i refers to the samples
                {
                    feature_values[i].feature = hypothesis->operator[](j).featureValue( *(allSamples->operator[](i)) );
                    feature_values[i].label   = allSamples->operator[](i)->getLabel();
                    feature_values[i].weight  = weight_distribution->operator[](i);

                    total_w_1_p += feature_values[i].weight * (feature_values[i].label == yes);
                    total_w_1_n += feature_values[i].weight * (feature_values[i].label == no);
                }

                std::sort( feature_values.begin(), feature_values.end() );

                weight_type total_w_0_p = 0; //sum of false negatives up to k. This is Viola and Jones' S+.
                weight_type total_w_0_n = 0; //sum of true negatives up to k.  This is Viola and Jones' S-.

                weight_type best_error = std::min(total_w_1_n, total_w_1_p);

                float v = feature_values[0].feature;
                Classification c0 = total_w_0_n <= total_w_0_p ? yes : no;
                Classification c1 = total_w_1_n <= total_w_1_p ? yes : no;

                for(WeightVector::size_type k = 0; k < feature_values.size(); ++k )
                {
                    total_w_0_p += feature_values[k].weight * (feature_values[k].label == yes);
                    total_w_0_n += feature_values[k].weight * (feature_values[k].label == no);

                    total_w_1_p -= feature_values[k].weight * (feature_values[k].label == yes);
                    total_w_1_n -= feature_values[k].weight * (feature_values[k].label == no);

                    if ( k < feature_values.size() - 1
                         && feature_values[k].feature == feature_values[k+1].feature )
                    {
                        continue;
                    }

                    //const weight_type error_o = std::min(total_w_0_n, total_w_0_p) + std::min(total_w_1_n, total_w_1_p); //Same as Viola and Jones'
                    const weight_type error = std::min(total_w_0_p + total_w_1_n,
                                                       total_w_0_n + total_w_1_p); //Viola and Jones' version.

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

                hypothesis->operator[](j).setThreshold(v);
                hypothesis->operator[](j).setPolarity(c0);

                {
                    //this must be synchonized
                    tbb::queuing_mutex::scoped_lock lock(*mutex);
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
                    tbb::queuing_mutex::scoped_lock lock(*mutex);
                    ++(*count);
                    progressCallback->tick(*count, hypothesis->size());
                    lock.release();
                }
            }
        }
    };



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
    Adaboost() : progressCallback(0),
                 weak_learner_mutex() {}

    Adaboost(ProgressCallback * progressCallback_) : progressCallback(progressCallback_),
                                                     weak_learner_mutex() {}

    virtual ~Adaboost() {}

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


        //Collects into allSamples pointers to both positive and negative LabeledExamples
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
                               DecisionStumpWeakLearner(&weak_learner_mutex,
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


            /*
#define DIRTY_DEBUG
#include <sstream>
            std::stringstream oss;
            oss << "file-" << t << ".csv";
            std::ofstream debugFile(oss.str().c_str());
            std::vector<weight_type> old(weight_distribution.size());
            std::copy(weight_distribution.begin(), weight_distribution.end(), old.begin());
#define DIRTY_DEBUG_END
            */

            //Now we just have to update the weight distribution of the samples.
            //Normalization factor is not inside the block because we report it to the progressCallback.
            const weight_type normalizationFactor =
                    updateWeightDistribution( allSamples,
                                              alpha,
                                              hypothesis[weak_hypothesis_index],
                                              weight_distribution );

            /*
#define DIRTY_DEBUG_2
            for( WeightVector::size_type i = 0; i < allSamples.size(); ++i )
            {
                debugFile << old[i] << ' ' << hypothesis[weak_hypothesis_index].classify( *(allSamples[i]) ) << ' ' << allSamples[i]->getLabel() << ' ' << weight_distribution[i] << std::endl;
            }
            debugFile.close();
#define DIRTY_DEBUG_2_END
            */


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

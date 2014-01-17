#ifndef DECISIONSTUMPWEAKLEARNER_H
#define DECISIONSTUMPWEAKLEARNER_H

#include <vector>
#include <tbb/tbb.h>

#include "common.h"
#include "labeledexample.h"
#include "progresscallback.h"



/**
 * Defines the mutex type a WeakLearner might use. It is recommended to define
 * this for improved maintainability.
 */
typedef tbb::queuing_mutex WeakLearnerMutex;



/**
 * This trait (http://www.cantrip.org/traits.html) allows the definition
 * of a WeakLearner template that works with both std::vector, tbb::blocked_range,
 * or any other container that conforms to the std::vector accessors interface.
 */
template<typename VectorType> struct VectorTraits;

template <>
struct VectorTraits< tbb::blocked_range<unsigned int> > {
    tbb::blocked_range<unsigned int> vector;
};

template <>
struct VectorTraits< std::vector<unsigned int> > {
    std::vector<unsigned int> vector;
};



/**
 * Abstract template implementation of a WeakLearner
 */
template <typename VectorType>
struct WeakLearner
{
    typedef VectorTraits<VectorType> vector_type;
    virtual void operator()(VectorType range) const =0;
};



/**
 * Implements a weak learner which weak classifiers are decision stumps.
 */
template <typename WeakHypothesisType>
class DecisionStumpWeakLearner// : WeakLearner< tbb::blocked_range<unsigned int> >
{
public:
    /**
     * @brief This class constructor.
     * @param mutex_
     * @param allSamples_
     * @param weight_distribution_
     * @param hypothesis_
     * @param selected_weak_hypothesis_weighted_error_
     * @param selected_weak_hypothesis_index_
     * @param count_
     * @param progressCallback_
     */
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
     * Runs this weak learner
     */
    void operator()(tbb::blocked_range< unsigned int > & range) const
    {
        //Feature values and respective weight and label
        std::vector<FeatureAndWeight> feature_values(allSamples->size());

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
            //Classification c1 = total_w_1_n <= total_w_1_p ? yes : no;

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
                    //c1 = total_w_1_n <= total_w_1_p ? yes : no;
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

            //synchronization needed only INSIDE the if
            if (progressCallback)
            {
                tbb::queuing_mutex::scoped_lock lock(*mutex);
                ++(*count);
                progressCallback->tick(*count, hypothesis->size());
                lock.release();
            }
        }
    }

private:

    struct FeatureAndWeight
    {
        float feature;
        weight_type weight;
        Classification label;

        bool operator < (const FeatureAndWeight & f) const
        {
            return feature < f.feature;
        }
    };

    WeakLearnerMutex                          * const mutex;
    const std::vector<const LabeledExample *> * const allSamples;
    const WeightVector                        * const weight_distribution;
    std::vector<WeakHypothesisType>           * const hypothesis;
    weight_type                               * const selected_weak_hypothesis_weighted_error;
    unsigned int                              * const selected_weak_hypothesis_index;
    unsigned long                             * const count;
    ProgressCallback                          * const progressCallback;
};



#endif // DECISIONSTUMPWEAKLEARNER_H

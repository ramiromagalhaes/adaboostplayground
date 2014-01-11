#ifndef PROGRESSCALLBACK_H
#define PROGRESSCALLBACK_H

#include "common.h"


/**
 * A callback to report the progress of the Adaboost train method. Just create
 * your implementation and pass an instance of it to the Adaboost constructor.
 */
struct ProgressCallback
{
    virtual ~ProgressCallback() =0;

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
    SimpleProgressCallback();

    virtual void beginAdaboostIteration(const unsigned int iteration);

    virtual void tick (const unsigned long current, const unsigned long total);

    virtual void classifierSelected (const weight_type alpha,
                                     const weight_type normalization_factor,
                                     const weight_type lowest_classifier_error,
                                     const unsigned int classifier_idx);
};



#endif // PROGRESSCALLBACK_H

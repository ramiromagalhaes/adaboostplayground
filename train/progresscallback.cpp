#include "progresscallback.h"

#include <iostream>



//http://stackoverflow.com/questions/8513408/c-abstract-base-class-constructors-destructors-general-correctness
ProgressCallback::~ProgressCallback() {} //All destructors must exist

SimpleProgressCallback::SimpleProgressCallback() : progress(-1) {}

void SimpleProgressCallback::beginAdaboostIteration(const unsigned int iteration)
{
    std::cout << "Adaboost iteration " << iteration << '\n';
    std::cout.flush();
}

void SimpleProgressCallback::tick (const unsigned long current, const unsigned long total)
{
    const int currentProgress = (int) (100 * current / total);
    if (currentProgress != progress)
    {
        progress = currentProgress;
        std::cout << "Progress: " << progress << "%.\r";
        std::cout.flush();
    }
}

void SimpleProgressCallback::classifierSelected (const weight_type alpha,
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

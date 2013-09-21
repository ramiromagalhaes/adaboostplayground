#ifndef TEMPLATE_TRAINCLASSIFIER_H
#define TEMPLATE_TRAINCLASSIFIER_H



#include <vector>
#include <iostream>

#include <opencv2/core/core.hpp>

#include "common.h"
#include "stronghypothesis.h"
#include "adaboost.h"
#include "haarclassifier.h"
#include "sampleextractor.h"



template<typename WeakHypothesisType>
int ___main(const std::string positivesFile,
           const std::string negativesFile,
           const std::string waveletsFile,
           const std::string strongHypothesisFile,
           const unsigned int maximum_iterations)
{
    StrongHypothesis<WeakHypothesisType> strongHypothesis(strongHypothesisFile);

    std::vector<LabeledExample> positiveSamples, negativeSamples;
    {
        if ( !SampleExtractor::fromImageFile(positivesFile, positiveSamples, yes) )
        {
            return 13;
        }
        std::cout << "Loaded " << positiveSamples.size() << " positive samples." << std::endl;

        //Viola and Jones state they used "6000 such non-face sub-windows" while building the cascade (2004, section 5.2).
        //On section 4.2 they show a different "simple experiment".
        if ( !SampleExtractor::extractRandomSample(6000, negativesFile, negativeSamples, no) )
        {
            return 17;
        }
        std::cout << "Loaded " << negativeSamples.size() << " negative samples." << std::endl;
    }

    std::vector<WeakHypothesisType> hypothesis;
    {
        loadHaarClassifiers(waveletsFile, hypothesis);
        std::cout << "Loaded " << hypothesis.size() << " weak classifiers." << std::endl;
    }

    Adaboost<WeakHypothesisType> boosting(new SimpleProgressCallback());

    try {
        boosting.train(positiveSamples,
                       negativeSamples,
                       strongHypothesis,
                       hypothesis,
                       maximum_iterations);
    } catch (int e) {
        std::cout << "Erro durante a execução do treinamento. Número do erro: " << e << std::endl;
    }

    return 0;
}



unsigned int charToInt(char * c)
{
    unsigned int i;
    std::stringstream(c) >> i;
    return i;
}



#endif // TEMPLATE_TRAINCLASSIFIER_H

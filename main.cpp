#include <vector>
#include <iostream>

#include <opencv2/core/core.hpp>

#include "common.h"
#include "dataprovider.h"
#include "stronghypothesis.h"
#include "adaboost.h"
#include "haarclassifier.h"
#include "sampleextractor.h"



unsigned int charToInt(char * c)
{
    std::stringstream ss;
    ss << c;
    unsigned int i;
    ss >> i;
    return i;
}



/**
 * Arguments:
 *     positivesIndexFile
 *     positivesImageFile
 *     negativesIndexFile
 *     negativesImageFile
 *     waveletsFile
 *     strongHypothesisOutputFile
 *     maximumIterations
 */
int main(int, char **argv) {
    const std::string positivesFile = argv[1];
    const std::string negativesFile = argv[2];
    const std::string waveletsFile = argv[3];
    const std::string strongHypothesisFile = argv[4];
    const unsigned int maximum_iterations = charToInt(argv[5]);

    StrongHypothesis<HaarClassifier> strongHypothesis(strongHypothesisFile);

    std::vector<LabeledExample> positiveSamples, negativeSamples;
    {
        if ( !SampleExtractor::fromIndexFile(positivesFile, positiveSamples, yes) )
        {
            return 13;
        }
        std::cout << "Loaded " << positiveSamples.size() << " positive samples." << std::endl;

        if ( !SampleExtractor::extractRandomSample(10000, negativesFile, negativeSamples, no) )
        {
            return 17;
        }
        std::cout << "Loaded " << negativeSamples.size() << " negative samples." << std::endl;
    }

    std::vector<HaarClassifier> hypothesis;
    {
        HaarClassifier::loadClassifiers(waveletsFile, hypothesis);
        std::cout << "Loaded " << hypothesis.size() << " weak classifiers." << std::endl;
    }

    Adaboost<HaarClassifier> boosting(new SimpleProgressCallback());

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

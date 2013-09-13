#include <vector>
#include <iostream>

#include <opencv2/core/core.hpp>

#include "common.h"
#include "dataprovider.h"
#include "stronghypothesis.h"
#include "adaboost.h"
#include "haarclassifier.h"



class MyProgressCallback : public WeakLearnerProgressCallback
{
private:
    int progress;

public:
    MyProgressCallback() : progress(-1) {}

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
        std::cout << "    Normalization factor: " << normalization_factor << std::endl;
    }
};



int main(int, char **argv) {
    const std::string positivesFile = argv[1];
    const std::string negativesFile = argv[2];
    const std::string waveletsFile = argv[3];
    const std::string strongHypothesisFile = argv[4];
    const unsigned int maximum_iterations = strtol(argv[5], 0, 10);

    StrongHypothesis<HaarClassifier> strongHypothesis(strongHypothesisFile);

    std::vector<HaarClassifier> hypothesis;
    {
        cv::Size * const size = new cv::Size(20, 20);
        HaarClassifier::loadClassifiers(size, waveletsFile, hypothesis);
        std::cout << "Loaded " << hypothesis.size() << " weak classifiers." << std::endl;
    }

    DataProvider provider(positivesFile, negativesFile);
    std::cout << "Total samples to load " << provider.size() << "." << std::endl;

    Adaboost<HaarClassifier> boosting(new MyProgressCallback());

    try {
        boosting.train(provider,
                       strongHypothesis,
                       hypothesis,
                       maximum_iterations);
    } catch (int e) {
        std::cout << "Erro durante a execução do treinamento. Número do erro: " << e << std::endl;
    }

    return 0;
}

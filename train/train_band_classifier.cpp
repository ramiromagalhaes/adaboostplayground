#include "template_trainclassifier.h"

/**
 * Trains the classifier proposed as new at A New Method for Haar-Like Features Weight
 * Adjustment Using Principal Component Analysis for Face Detection
 * (http://www.thinkmind.org/index.php?view=article&articleid=icons_2014_3_20_40057)
 *
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
    const std::string negativesIndexFile = argv[3];
    const std::string waveletsFile = argv[4];
    const std::string strongHypothesisFile = argv[5];
    const unsigned int maximum_iterations = charToInt(argv[6]);

    ___main<MyHaarClassifier, DecisionStumpWeakLearner<MyHaarClassifier> >(
                positivesFile,
                negativesFile,
                negativesIndexFile,
                waveletsFile,
                strongHypothesisFile,
                maximum_iterations);
}

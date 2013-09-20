#include "prototype_trainclassifier.h"

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

    ___main<HaarClassifier>(
                positivesFile,
                negativesFile,
                waveletsFile,
                strongHypothesisFile,
                maximum_iterations);
}

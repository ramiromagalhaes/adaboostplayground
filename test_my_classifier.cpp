#include "template_testclassifier.h"



/**
 * Arguments:
 *     positivesIndexFile
 *     positivesImageFile
 *     negativesIndexFile
 *     negativesImageFile
 *     strongHypothesisInputFile
 */
int main(int, char **argv) {
    const std::string testImagesIndexFileName = argv[1];
    const std::string groundTruthFileName = argv[2];
    const std::string strongHypothesisFile = argv[3];

    return ___main<MyHaarClassifier>(testImagesIndexFileName,
                                     groundTruthFileName,
                                     strongHypothesisFile);
}

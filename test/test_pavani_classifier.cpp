#include "template_testclassifier.h"



/**
 *
 */
int main(int, char **argv) {
    const std::string testImagesIndexFileName = argv[1];
    const std::string groundTruthFileName = argv[2];
    const std::string strongHypothesisFile = argv[3];
    const std::string rocCurveFile = argv[4];

    return ___main<PavaniHaarClassifier>(
                testImagesIndexFileName,
                groundTruthFileName,
                strongHypothesisFile,
                rocCurveFile);
}

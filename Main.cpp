#include <vector>
#include <iostream>

#include <boost/filesystem.hpp>

#include "Common.h"
#include "WeakLearner.h"
#include "StrongHypothesis.h"
#include "Adaboost.h"
#include "Haarwavelet.h"
#include "Haarwaveletutilities.h"
#include "dataprovider.h"



class HaarClassifier : public WeakHypothesis, public HaarWavelet
{
public:
    int p;
    float theta;

    HaarClassifier() : p(1), theta(.0f) { }

    ~HaarClassifier() { }

    Classification classify(const cv::Mat &data) const //TODO who should pass the data to the wavelet?
    {
        if (p > 0)
        {
            return value() > theta ? yes : no;
        }
        return value() < theta ? yes : no;
    }

    bool write(std::ostream & output) //TODO is this really necessary???
    {
        HaarWavelet::write(output);
    }
};



int main(int argc, char **argv) {
    const unsigned int buffer_size = strtol(argv[1], 0, 10);
    boost::filesystem::path positivesFile = "/mnt/faces.txt";
    boost::filesystem::path negativesFile = "/mnt/background.txt";

    const unsigned int maximum_iterations = 40;

    StrongHypothesis strongHypothesis("/mnt/strongHypothesis.txt");

    std::vector < HaarClassifier * > * hypothesis = new std::vector < HaarClassifier * >();
    {
        cv::Size size = cv::Size(20, 20);
        loadHaarWavelets(&size,
                         "/home/ramiro/workspace/ecrsgen/data/haarwavelets.txt",
                         *((std::vector < HaarWavelet * > *)hypothesis) );
        std::cout << "Loaded " << hypothesis->size() << " weak classifiers.\n";
    }

    DataProvider provider(positivesFile, negativesFile, buffer_size);

    Adaboost boosting;

    return 0;

    try {
        boosting.train(provider,
                       strongHypothesis,
                       *(std::vector < WeakHypothesis * > *)hypothesis,
                       maximum_iterations);
    } catch (int e) {
        std::cout << "Erro durante a execução do treinamento. Número do erro: " << e << std::endl;
    }

    return 0;
}

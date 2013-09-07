#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <boost/filesystem.hpp>

#include "Common.h"
#include "StrongHypothesis.h"
#include "Adaboost.h"
#include "Haarwavelet.h"
#include "dataprovider.h"



class HaarClassifier : public WeakHypothesis
{
public:
    int p;
    float theta;

    HaarWavelet * wavelet;

    HaarClassifier() : p(1), theta(.0f) { }
    HaarClassifier(HaarWavelet * const wavelet_) : p(1), theta(.0f), wavelet(wavelet_) { }

    virtual ~HaarClassifier() {
        delete wavelet;
    }

    virtual Classification classify(const cv::Mat & sample) const
    {
        //TODO who should pass the data to the wavelet?
        //TODO how should I produce the the integral image? In here? Out of here? If out, then what parameter should I pass?
        //TODO this is a temporary stupid solution for testing purposes.
        cv::Mat integralSum(sample.rows + 1, sample.cols + 1, CV_32S);
        cv::Mat integralSquare(sample.rows + 1, sample.cols + 1, CV_32S);
        cv::integral(sample, integralSum, integralSquare, CV_32S);
        wavelet->setIntegralImages(&integralSum, &integralSquare);

        if (p > 0)
        {
            return wavelet->value() > theta ? yes : no;
        }
        return wavelet->value() < theta ? yes : no;
    }

    virtual bool write(std::ostream & output) const
    {
        return wavelet->write(output);
    }
};



bool loadClassifiers(cv::Size * const sampleSize, const std::string &filename, std::vector<HaarClassifier *> & wavelets)
{
    std::ifstream ifs;
    ifs.open(filename.c_str(), std::ifstream::in);

    if ( !ifs.is_open() )
    {
        return false;
    }

    do
    {
        HaarClassifier * classifier = new HaarClassifier(new HaarWavelet(sampleSize, ifs));
        if ( !ifs.eof() )
        {
            wavelets.push_back(classifier);
        }
        else
        {
            break;
        }
    } while (true);

    ifs.close();

    return true;
}



int main(int argc, char **argv) {
    const unsigned int buffer_size = strtol(argv[1], 0, 10);
    boost::filesystem::path positivesFile = "/mnt/faces.txt";
    boost::filesystem::path negativesFile = "/mnt/background-excerpt.txt";

    const unsigned int maximum_iterations = 40;

    StrongHypothesis strongHypothesis("/mnt/strongHypothesis.txt");
    std::cout << "Strong hypothesis ready." << std::endl;

    std::vector < HaarClassifier * > * hypothesis = new std::vector < HaarClassifier * >();
    {
        cv::Size size = cv::Size(20, 20);
        loadClassifiers(&size,
                        "/home/ramiro/workspace/ecrsgen/data/haarwavelets.txt",
                        *((std::vector < HaarClassifier * > *)hypothesis) );
        std::cout << "Loaded " << hypothesis->size() << " weak classifiers." << std::endl;
    }

    DataProvider provider(positivesFile, negativesFile, buffer_size);
    std::cout << "Data provider has " << provider.size() << " samples." << std::endl;

    Adaboost boosting;
    std::cout << "Boosting object created." << std::endl;

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

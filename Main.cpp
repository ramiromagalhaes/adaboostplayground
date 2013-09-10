#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>

#include "Haarwavelet.h"
#include "Common.h"
#include "dataprovider.h"
#include "StrongHypothesis.h"
#include "Adaboost.h"



class HaarClassifier
{
public:
    int p;
    float theta;
    HaarWavelet * wavelet;

    cv::Mat * lastSample;   //Holds a pointer to the last sample provided to the classifier.
    cv::Mat integralSum;    //If lastSample changes, we'll need another integralSum...
    cv::Mat integralSquare; //...and another integralSquare.

    HaarClassifier() : p(1),
                       theta(.0f),
                       wavelet(0),
                       lastSample(0),
                       integralSum(21, 21, CV_64F),
                       integralSquare(21, 21, CV_64F) { }

    HaarClassifier(HaarWavelet * w) : p(1),
                                      theta(.0f),
                                      wavelet(w),
                                      lastSample(0),
                                      integralSum(21, 21, CV_64F),
                                      integralSquare(21, 21, CV_64F)
    {
        wavelet->setIntegralImages(&integralSum, &integralSquare);
    }

    //http://pages.cs.wisc.edu/~hasti/cs368/CppTutorial/NOTES/CLASSES-PTRS.html#destructor
    //http://stackoverflow.com/questions/6435404/c-error-double-free-or-corruption-fasttop

    HaarClassifier(const HaarClassifier & h) : p(h.p),
                                               theta(h.theta),
                                               wavelet(h.wavelet),
                                               lastSample(h.lastSample),
                                               integralSum(21, 21, CV_64F),
                                               integralSquare(21, 21, CV_64F)
    {
        wavelet->setIntegralImages(&integralSum, &integralSquare);
    }



    HaarClassifier &operator=(const HaarClassifier & h)
    {
        p = h.p;
        theta = h.theta;
        wavelet = h.wavelet;
        lastSample = h.lastSample;
        integralSum = h.integralSum;
        integralSquare = h.integralSquare;
        wavelet->setIntegralImages(&integralSum, &integralSquare);
        return *this;
    }

    ~HaarClassifier() {
        /* TODO properly delete the reference to wavelet. It is currently not working.
        if (wavelet != 0)
        {
            delete wavelet;
            wavelet = 0;
        }
        */
    }

    Classification classify(const cv::Mat & sample) const
    {
        //TODO who should pass the data to the wavelet?
        //TODO how should I produce the the integral image? In here? Out of here? If out, then what parameter should I pass?
        //TODO this is a temporary solution for testing purposes.
        //if ( lastSample != &sample )
        {
            cv::integral(sample, integralSum, integralSquare, CV_64F);
        }

        if (p > 0)
        {
            return wavelet->value() > theta ? yes : no;
        }
        return wavelet->value() < theta ? yes : no;
    }

    bool write(std::ostream & output) const
    {
        return wavelet->write(output);
    }
};



bool loadClassifiers(const std::string &filename, std::vector<HaarClassifier> & classifiers)
{
    cv::Size * const size = new cv::Size(20, 20);

    std::ifstream ifs;
    ifs.open(filename.c_str(), std::ifstream::in);

    if ( !ifs.is_open() )
    {
        return false;
    }

    do
    {
        HaarClassifier classifier(new HaarWavelet(size, ifs));

        if ( !ifs.eof() )
        {
            classifiers.push_back( classifier );
        }
        else
        {
            break;
        }
    } while (true);

    ifs.close();

    return true;
}



int main(int, char **argv) {
    const std::string positivesFile = argv[1];
    const std::string negativesFile = argv[2];
    const std::string waveletsFile = argv[3];
    const std::string strongHypothesisFile = argv[4];
    const unsigned int maximum_iterations = strtol(argv[5], 0, 10);

    StrongHypothesis<HaarClassifier> strongHypothesis(strongHypothesisFile);

    std::vector<HaarClassifier> hypothesis;
    {
        loadClassifiers(waveletsFile, hypothesis);
        std::cout << "Loaded " << hypothesis.size() << " weak classifiers." << std::endl;
    }

    DataProvider provider(positivesFile, negativesFile);
    std::cout << "Total samples to load " << provider.size() << "." << std::endl;

    Adaboost<HaarClassifier> boosting;

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

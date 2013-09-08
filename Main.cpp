#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <boost/filesystem.hpp>

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

    cv::Mat integralSum;
    cv::Mat integralSquare;

    HaarClassifier() : p(1),
                       theta(.0f),
                       integralSum(21, 21, CV_64F),
                       integralSquare(21, 21, CV_64F),
                       wavelet(0) { }

    HaarClassifier(HaarWavelet * w) : p(1),
                                      theta(.0f),
                                      integralSum(21, 21, CV_64F),
                                      integralSquare(21, 21, CV_64F),
                                      wavelet(w)
    {
        wavelet->setIntegralImages(&integralSum, &integralSquare);
    }

    //http://pages.cs.wisc.edu/~hasti/cs368/CppTutorial/NOTES/CLASSES-PTRS.html#destructor
    //http://stackoverflow.com/questions/6435404/c-error-double-free-or-corruption-fasttop

    HaarClassifier(const HaarClassifier & h) : p(h.p),
                                               theta(h.theta),
                                               wavelet(h.wavelet),
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
        //TODO this is a temporary stupid solution for testing purposes.
        cv::integral(sample, integralSum, integralSquare, CV_64F);

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
    const unsigned int buffer_size = strtol(argv[1], 0, 10);
    std::string positivesFile = "/mnt/faces.txt";
    std::string negativesFile = "/mnt/background-excerpt.txt";

    const unsigned int maximum_iterations = 40;

    StrongHypothesis<HaarClassifier> strongHypothesis("/mnt/strongHypothesis.txt");

    std::vector<HaarClassifier> hypothesis;
    {
        loadClassifiers("/mnt/haarwavelets.txt", hypothesis);
        std::cout << "Loaded " << hypothesis.size() << " weak classifiers." << std::endl;
    }

    DataProvider provider(positivesFile, negativesFile, buffer_size);
    std::cout << "Data provider has " << provider.size() << " samples." << std::endl;

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

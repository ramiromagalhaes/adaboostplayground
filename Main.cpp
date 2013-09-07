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

    HaarClassifier()
    {
        p = 1;
        theta = .0f;
        wavelet = 0;
    }

    HaarClassifier(HaarWavelet * w)
    {
        p = 1;
        theta = .0f;
        wavelet = w;
    }

    //http://pages.cs.wisc.edu/~hasti/cs368/CppTutorial/NOTES/CLASSES-PTRS.html#destructor
    //http://stackoverflow.com/questions/6435404/c-error-double-free-or-corruption-fasttop

    HaarClassifier(const HaarClassifier & h) : p(h.p),
                                               theta(h.theta),
                                               wavelet(h.wavelet) { }



    HaarClassifier &operator=(const HaarClassifier & h)
    {
        p = h.p;
        theta = h.theta;
        wavelet = h.wavelet;
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
        if ( !ifs.eof() )
        {
            HaarWavelet * wavelet = new HaarWavelet(size, ifs);
            HaarClassifier classifier;
            classifier.wavelet = wavelet;
            classifiers.push_back(classifier);
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
    std::cout << "Strong hypothesis ready." << std::endl;

    std::vector<HaarClassifier> hypothesis;
    {
        loadClassifiers("/mnt/haarwavelets.txt", hypothesis);
        std::cout << "Loaded " << hypothesis.size() << " weak classifiers." << std::endl;
    }

    DataProvider provider(positivesFile, negativesFile, buffer_size);
    std::cout << "Data provider has " << provider.size() << " samples." << std::endl;

    Adaboost<HaarClassifier> boosting;
    std::cout << "Boosting object created." << std::endl;

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

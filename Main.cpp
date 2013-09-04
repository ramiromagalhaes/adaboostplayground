#include <limits>
#include <vector>
#include <iostream>
#include <sstream> //for std::stringstream
#include <string>
#include <ostream>
#include <cstdlib>

#include "Common.h"
#include "WeakLearner.h"
#include "StrongHypothesis.h"
#include "Adaboost.h"



int main(int argc, char **argv) {
    if (argc != 2)
    {
        return -1;
    }
    const unsigned int maximum_iterations = strtol(argv[1], 0, 10);

    std::vector < LabeledExample > training_set; //TODO load from somewhere
    std::vector < WeakHypothesis * > hypothesis; //TODO load from somewhere

    //WeakLearner * learner = new MyWeakLearner();
    StrongHypothesis strong_hypothesis;
    Adaboost boosting; //default constructor uses the ReweightingWeakLearner

    try {
        boosting.train(training_set, strong_hypothesis, hypothesis, maximum_iterations);
    } catch (int e) {
        std::cout << "Erro durante a execução do treinamento. Número do erro: " << e << std::endl;
    }

    int false_positive = 0;
    int true_negative = 0; //positive samples that were classified as negative.

    for (std::vector < LabeledExample >::const_iterator it = training_set.begin(); it != training_set.end(); ++it) {
        const Classification c = strong_hypothesis.classify(it->example);
        if (it->label == yes) {
            if (c == no) {
                true_negative++;
            }
        } else /* le->label == no*/ {
            if (c == yes) {
                false_positive++;
            }
        }

        std::cout << it->example << " esperado " << it->label << " obtido " << c << std::endl;
    }

    std::cout << "True negative        : " << true_negative  / (double)training_set.size() << std::endl;
    std::cout << "False positive       : " << false_positive / (double)training_set.size() << std::endl;
    std::cout << "Misclassified        : " << (true_negative + false_positive) / (double)training_set.size() << std::endl;
    std::cout << "Correctly classified : " << (training_set.size() - true_negative - false_positive) / (double)training_set.size() << std::endl;

    return 0;
}

#ifndef STRONGHYPOTHESIS_H_
#define STRONGHYPOTHESIS_H_

#include <vector>
#include "Common.h"



/**
 * Instances of this class hold the weights and weak classifiers that make the strong classifier.
 * Notice that this class holds pointers to the weak classifiers, added via the insert method.
 * All those pointers will be deleted on destruction of this class.
 */
class StrongHypothesis {
private:

    /**
     * @brief The entry class holds the WeakHypothesis pointer and it's weight.
     */
    class entry {
    public:
        weight_type weight;
        WeakHypothesis const * weakHypothesis;

        entry() : weight(0), weakHypothesis(0) {}
        entry(weight_type w, WeakHypothesis const * h) : weight(w),
                                                         weakHypothesis(h) {

        }
        ~entry() {
            delete weakHypothesis;
        }
    };

    std::vector<entry> hypothesis;

public:
    //note: intentional inline methods thanks to templates. See http://stackoverflow.com/questions/644397/c-class-with-template-cannot-find-its-constructor
    StrongHypothesis() {}

     ~StrongHypothesis() {
        hypothesis.clear();
    }



    void insert(weight_type alpha, WeakHypothesis const * const weak_hypothesis) {
        entry e(alpha, const_cast<WeakHypothesis const *>(weak_hypothesis)); //from now on, this class is responsible for handling properly these pointers
        hypothesis.insert(hypothesis.end(), e);
    }



    Classification classify(const cv::Mat &input) const {
        weight_type result = .0f;

        for (std::vector<entry>::const_iterator it = hypothesis.begin(); it != hypothesis.end(); ++it) {
            entry e = *it;
            result += (e.weight) * (e.weakHypothesis->classify(input));
        }

        return result >= 0 ? yes : no;
    }


    //TODO accessor methods to the weak hypothesis
};



#endif /* STRONGHYPOTHESIS_H_ */

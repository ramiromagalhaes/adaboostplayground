#ifndef WEAKHYPOTHESIS_H_
#define WEAKHYPOTHESIS_H_

#include "Common.h"

class WeakHypothesis {
public:
    WeakHypothesis() { }
    virtual ~WeakHypothesis() { }
    virtual Classification classify(const cv::Mat &data) const =0;
};

#endif /* WEAKHYPOTHESIS_H_ */

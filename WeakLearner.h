/*
 * WeakLearner.h
 *
 *  Created on: 06/03/2013
 *      Author: ramiro
 */

#ifndef WEAKLEARNER_H_
#define WEAKLEARNER_H_

#include "WeakHypothesis.h"
#include <vector>

template <class dataType> class WeakLearner {
public:
	WeakLearner();
	virtual ~WeakLearner();

	WeakHypothesis learn(std::vector<dataType> &data);
};

#endif /* WEAKLEARNER_H_ */

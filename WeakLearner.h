/*
 * WeakLearner.h
 *
 *  Created on: 06/03/2013
 *      Author: ramiro
 */

#ifndef WEAKLEARNER_H_
#define WEAKLEARNER_H_

#include <vector>
#include <utility>
#include "WeakHypothesis.h"

template <typename dataType> class WeakLearner {
public:
	WeakLearner() {} //note: intentional inline con/destructor. See http://stackoverflow.com/questions/644397/c-class-with-template-cannot-find-its-constructor
	virtual ~WeakLearner() {}

	WeakHypothesis<dataType> learn(const std::vector< std::pair<dataType, Classification> > &data) {
		// TODO Auto-generated method stub
		return WeakHypothesis<dataType>();
	}

};

#endif /* WEAKLEARNER_H_ */

/*
 * Adaboost.h
 *
 *  Created on: 02/03/2013
 *      Author: ramiro
 */

#ifndef ADABOOST_H_
#define ADABOOST_H_

#include <vector>
#include <utility>
#include "StrongHypothesis.h"



enum Classification {
	no = -1, yes = 1
};



template<class dataType> class Adaboost {
public:
	Adaboost();
	virtual ~Adaboost();

	//probably should return a pointer
	StrongHypothesis train(const std::vector< std::pair<dataType, Classification> > &trainingData);

private:
	int t; //iteration (epoch) counter

	void init_distribution(std::vector<double> &distribution);
};


#endif /* ADABOOST_H_ */

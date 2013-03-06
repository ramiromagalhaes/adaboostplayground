/*
 * Data.h
 *
 *  Created on: 03/03/2013
 *      Author: ramiro
 */

#ifndef DATA_H_
#define DATA_H_

#include "Adaboost.h"

class Data {
public:
	Data();

	std::vector<Hypothesis> hypothesisDatabase;
	std::vector<Point> pointDatabase;

	void load();
};

#endif /* DATA_H_ */

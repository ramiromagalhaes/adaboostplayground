/*
 * Data.cpp
 *
 *  Created on: 03/03/2013
 *      Author: ramiro
 */

#include "Data.h"

Data::Data() {
}

void Data::load() {
	hypothesisDatabase.insert(hypothesisDatabase.end(), Hypothesis(vertical, 1.0));
	hypothesisDatabase.insert(hypothesisDatabase.end(), Hypothesis(vertical, 3.0));
	hypothesisDatabase.insert(hypothesisDatabase.end(), Hypothesis(vertical, 5.0));
	hypothesisDatabase.insert(hypothesisDatabase.end(), Hypothesis(vertical, 7.0));
	hypothesisDatabase.insert(hypothesisDatabase.end(), Hypothesis(vertical, 9.0));
	hypothesisDatabase.insert(hypothesisDatabase.end(), Hypothesis(horizontal, 1.0));
	hypothesisDatabase.insert(hypothesisDatabase.end(), Hypothesis(horizontal, 3.0));
	hypothesisDatabase.insert(hypothesisDatabase.end(), Hypothesis(horizontal, 5.0));
	hypothesisDatabase.insert(hypothesisDatabase.end(), Hypothesis(horizontal, 7.0));
	hypothesisDatabase.insert(hypothesisDatabase.end(), Hypothesis(horizontal, 9.0));

	pointDatabase.insert(pointDatabase.end(), Point(1, 1));
	pointDatabase.insert(pointDatabase.end(), Point(2, 2));
	pointDatabase.insert(pointDatabase.end(), Point(3, 4));
	pointDatabase.insert(pointDatabase.end(), Point(4, 3));
	pointDatabase.insert(pointDatabase.end(), Point(6, 3));
	pointDatabase.insert(pointDatabase.end(), Point(1, 5));
	pointDatabase.insert(pointDatabase.end(), Point(2, 8));
	pointDatabase.insert(pointDatabase.end(), Point(3, 5));
	pointDatabase.insert(pointDatabase.end(), Point(7, 9));
	pointDatabase.insert(pointDatabase.end(), Point(9, 2));
}

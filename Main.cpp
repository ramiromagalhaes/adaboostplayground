/*
 * Main.cpp
 *
 *  Created on: 02/03/2013
 *      Author: ramiro
 */



#include <vector>
#include <utility>
#include "WeakLearner.h"
#include "StrongHypothesis.h"
#include "Adaboost.h"



class Point {
public:
	double x, y;

	Point () {
		x = y = 0;
	}

	Point(double x, double y) {
		this->x = x;
		this->y = y;
	}

	virtual ~Point() {}
};



enum Orientation {
	vertical, horizontal
};



/*
class Hypothesis {
public:
	Hypothesis(Orientation o, double p) {
		orientation = o;
		position = p;
	}

	Classification test(const Point &input) {
		switch (orientation) {
			case vertical:
				//if input is to the left of the vertical bar, then return yes
				if (input.x <= position) {
					return yes;
				} else {
					return no;
				}
			case horizontal:
				//if input is above the horizontal bar, then return yes
				if (input.y >= position) {
					return yes;
				} else {
					return no;
				}
			default:
				throw "!!!";
		}
	}

private:
	double position;
	Orientation orientation;
};
*/



int main(int argc, char **argv) {
	//TODO load training data from somewhere
	typename Adaboost<Point>::traning_data_type training_data;

	WeakLearner<Point> learner; //TODO implement a real Weak Learner

	StrongHypothesis<Point> strong_hypothesis;

	Adaboost<Point> boosting(learner);
	boosting.train(training_data, strong_hypothesis);

	return 0;
}

/*
 * Main.cpp
 *
 *  Created on: 02/03/2013
 *      Author: ramiro
 */

#include "Adaboost.h"


enum Orientation {
	vertical, horizontal
};



class Point {
public:
	double x, y;
	Point(double x, double y) {
		this->x = x;
		this->y = y;
	}
};



class Hypothesis {
public:
	Hypothesis(Orientation o, double p) {
		orientation = o;
		position = p;
	}

	Classification test(const Point input) {
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





int main(int argc, char **argv) {

	return 0;
}

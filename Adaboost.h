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



enum Classification {
	no = -1, yes = 1
};



enum Orientation {
	vertical, horizontal
};



class Point {
public:
	double first, second;
	Point(double x, double y);
};



class Hypothesis {
public:
	Hypothesis(Orientation o, double p);
	Classification test(const Point input);
private:
	double position;
	Orientation orientation;
};



class Adaboost {
public:
	Adaboost();
	virtual ~Adaboost();
	void train();

private:
	std::vector< std::pair<Point, Classification> > trainingData;
	std::vector<double> distribution;//holds all M elements in the last t

	int t;
	//holds t-related stuff
	std::vector<double> alpha; //alpha subscript t where t is the vector's index
	std::vector<Hypothesis> finalHypothesis;
	std::vector<double> weightedError;

	Hypothesis get_weak_hypothesis();
	void choose_apha();
	void update_distribution(Hypothesis& currentWeakHypothesis);
};


#endif /* ADABOOST_H_ */

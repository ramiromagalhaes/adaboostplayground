/*
 * Main.cpp
 *
 *  Created on: 02/03/2013
 *      Author: ramiro
 */



#include <limits>
#include <vector>
#include <iostream>
#include <sstream> //for std::stringstream
#include <string>
#include <ostream>

#include "Common.h"
#include "WeakLearner.h"
#include "StrongHypothesis.h"
#include "Adaboost.h"



enum Orientation {
	vertical, horizontal
};



class Point {
public:
	double x, y;

	Point () : x(0), y(0) {}
	Point(double _x, double _y) : x(_x), y(_y) {}
	virtual ~Point() {}

	friend std::ostream& operator<<(std::ostream& os, Point& s) {
		os << "Point(" << s.x << ", " << s.y << ")";
		return os;
	}

};



class MyWeakHypothesis : public WeakHypothesis<Point> {
public:
	Orientation orientation;
	double position;

	MyWeakHypothesis(Orientation o, double p) : WeakHypothesis<Point>() {
		orientation = o;
		position = p;
	}

	virtual Classification classify(const Point &input) const {
		switch (orientation) {
			case vertical  : return input.x < position ? yes : no;
			case horizontal: return input.y < position ? yes : no;
			default        : throw 1;
		}
	}

	virtual std::string str() const {
		std::string os("Hypothesis ");
		switch (orientation) {
			case vertical  : os.append("V "); break;
			case horizontal: os.append("H "); break;
			default        : throw 3;
		}

		{
			std::ostringstream oss;
			oss << position;
			os.append(oss.str());
		}

		os.append("; ");
		return os;
	}
};



class MyWeakLearner : public WeakLearner<Point> {
private:
	std::vector<MyWeakHypothesis*> hypothesis;

public:
	MyWeakLearner() : WeakLearner<Point>() {
		MyWeakHypothesis *h = new MyWeakHypothesis(vertical, 3);
		hypothesis.insert(hypothesis.end(), h);

		h = new MyWeakHypothesis(vertical, 5);
		hypothesis.insert(hypothesis.end(), h);

		h = new MyWeakHypothesis(vertical, 7);
		hypothesis.insert(hypothesis.end(), h);

		h = new MyWeakHypothesis(vertical, 9);
		hypothesis.insert(hypothesis.end(), h);

		h = new MyWeakHypothesis(horizontal, 2);
		hypothesis.insert(hypothesis.end(), h);

		h = new MyWeakHypothesis(horizontal, 4);
		hypothesis.insert(hypothesis.end(), h);

		h = new MyWeakHypothesis(horizontal, 6);
		hypothesis.insert(hypothesis.end(), h);

		h = new MyWeakHypothesis(horizontal, 8);
		hypothesis.insert(hypothesis.end(), h);
	};

	~MyWeakLearner(){};

	virtual WeakHypothesis<Point>* learn(
			const std::vector < LabeledExample<Point> * > &training_sample,
			const std::vector < double > &weighted_errors,
			double &weighted_error) {

		double lowest_error = std::numeric_limits<double>::max();
		std::vector < LabeledExample<Point> >::size_type best_hypothesis_index = 0;

		for (std::vector < LabeledExample<Point> >::size_type i = 0; i < hypothesis.size(); i++) {
			MyWeakHypothesis const * const hyp = hypothesis[i];

			weighted_error = 0;

			for (typename std::vector < LabeledExample<Point> * >::const_iterator itTrain = training_sample.begin(); itTrain != training_sample.end(); ++itTrain) {
				LabeledExample<Point> const * const train = *itTrain;
				if (hyp->classify(train->example) != train->label) {
					weighted_error += weighted_errors[i];
				}
			}

			if (weighted_error < lowest_error) {
				lowest_error = weighted_error;
				best_hypothesis_index = i;
			}
		}

		weighted_error = lowest_error;

		return hypothesis[best_hypothesis_index];
	}
};



std::vector < LabeledExample <Point> * >* get_training_data() {
	std::vector < LabeledExample <Point> * > * const data = new std::vector < LabeledExample <Point> * >();

	Point *p = new Point(1, 2);
	LabeledExample<Point> * d = new LabeledExample<Point>(*p, yes);
	data->push_back(d);

	p = new Point(2, 7);
	d = new LabeledExample<Point>(*p, yes);
	data->push_back(d);

	p = new Point(3, 5);
	d = new LabeledExample<Point>(*p, yes);
	data->push_back(d);

	p = new Point(4, 4);
	d = new LabeledExample<Point>(*p, yes);
	data->push_back(d);

	p = new Point(5, 6);
	d = new LabeledExample<Point>(*p, yes);
	data->push_back(d);

	p = new Point(6, 2);
	d = new LabeledExample<Point>(*p, yes);
	data->push_back(d);

	p = new Point(6, 9);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(8, 5);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(9, 7);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(8, 3);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(9, 1);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	return data;
}

int main(int argc, char **argv) {
	try {
		std::vector < LabeledExample <Point> * > const * const training_set = get_training_data();

		WeakLearner<Point> *learner = new MyWeakLearner();
		StrongHypothesis<Point> strong_hypothesis;
		Adaboost<Point> boosting(learner);

		boosting.train(*training_set, strong_hypothesis);

		std::cout << strong_hypothesis;
		std::cout << std::endl;

		for (std::vector < LabeledExample <Point> * >::const_iterator it = training_set->begin(); it != training_set->end(); ++it) {
			LabeledExample<Point> * le = *it;
			std::cout << le->example << " esperado " << le->label << " obtido " << strong_hypothesis.classify(le->example) << std::endl;
		}
	} catch (int e) {
		std::cout << "Erro " << e << std::endl;
	}

	return 0;
}

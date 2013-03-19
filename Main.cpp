/*
 * Main.cpp
 *
 *  Created on: 02/03/2013
 *      Author: ramiro
 */



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

	virtual std::string toString() const {
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

	virtual WeakHypothesis<Point>* learn(const std::vector < training_data<Point> * > &data) {
		double lowest_error = 1;
		std::vector < training_data<Point> >::size_type best_hypothesis_index = 0;

		for (std::vector < training_data<Point> >::size_type i = 0; i < hypothesis.size(); i++) {
			MyWeakHypothesis const * const hyp = hypothesis[i];
			int wrong_conclusions = 0;

			for (typename std::vector < training_data<Point> * >::const_iterator itTrain = data.begin(); itTrain != data.end(); ++itTrain) {
				training_data<Point> const * const train = *itTrain;
				if (hyp->classify(train->data) != train->classification) {
					wrong_conclusions++;
				}
			}

			const double error_ratio = ((double)wrong_conclusions) / ((double)data.size());

			if (error_ratio < lowest_error) {
				lowest_error = error_ratio;
				best_hypothesis_index = i;
			}
		}

		return hypothesis[best_hypothesis_index];
	}
};



std::vector < training_data <Point> * >* get_training_data() {
	std::vector < training_data <Point> * > * const data = new std::vector < training_data <Point> * >();

	Point *p = new Point(1, 2);
	training_data<Point> * d = new training_data<Point>(*p, yes);
	data->push_back(d);

	p = new Point(2, 7);
	d = new training_data<Point>(*p, yes);
	data->push_back(d);

	p = new Point(3, 5);
	d = new training_data<Point>(*p, yes);
	data->push_back(d);

	p = new Point(4, 4);
	d = new training_data<Point>(*p, yes);
	data->push_back(d);

	p = new Point(5, 6);
	d = new training_data<Point>(*p, yes);
	data->push_back(d);

	p = new Point(6, 2);
	d = new training_data<Point>(*p, yes);
	data->push_back(d);

	p = new Point(6, 9);
	d = new training_data<Point>(*p, no);
	data->push_back(d);

	p = new Point(8, 5);
	d = new training_data<Point>(*p, no);
	data->push_back(d);

	p = new Point(9, 7);
	d = new training_data<Point>(*p, no);
	data->push_back(d);

	p = new Point(8, 3);
	d = new training_data<Point>(*p, no);
	data->push_back(d);

	p = new Point(9, 1);
	d = new training_data<Point>(*p, no);
	data->push_back(d);

	return data;
}

int main(int argc, char **argv) {
	try {
		std::vector < training_data <Point> * > const * const training_data = get_training_data();

		WeakLearner<Point> *learner = new MyWeakLearner();
		StrongHypothesis<Point> strong_hypothesis;
		Adaboost<Point> boosting(learner);

		boosting.train(*training_data, strong_hypothesis);

		std::cout << strong_hypothesis;
		std::cout << std::endl;

		{
			Point p(2, 8);
			std::cout << strong_hypothesis.classify(p) << std::endl;
		}
		{
			Point p(6, 6);
			std::cout << strong_hypothesis.classify(p) << std::endl;
		}
		{
			Point p(1, 1);
			std::cout << strong_hypothesis.classify(p) << std::endl;
		}
		{
			Point p(7, 9);
			std::cout << strong_hypothesis.classify(p) << std::endl;
		}
		{
			Point p(8, 3);
			std::cout << strong_hypothesis.classify(p) << std::endl;
		}
	} catch (int e) {
		std::cout << "Erro " << e << std::endl;
	}

	return 0;
}

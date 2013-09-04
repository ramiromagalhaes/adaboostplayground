#include <limits>
#include <vector>
#include <iostream>
#include <sstream> //for std::stringstream
#include <string>
#include <ostream>
#include <cstdlib>

#include "Common.h"
#include "WeakLearner.h"
#include "StrongHypothesis.h"
#include "Adaboost.h"



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
    enum Orientation {
        vertical = -1, horizontal = 1
    };

    enum Inclusion {
        before = -1, after = 1
    };

    Orientation orientation;
    Inclusion inclusion;
    double position;

    MyWeakHypothesis(Orientation o, Inclusion i, double p) : WeakHypothesis<Point>() {
        //orientation(o), inclusion(i), position(p)
        orientation = o;
        inclusion = i;
        position = p;
    }

    virtual Classification classify(const Point &input) const {
        switch (orientation) {
            case vertical  : return inclusion * (input.x - position) >= 0 ? yes : no;
            case horizontal: return inclusion * (input.y - position) >= 0 ? yes : no;
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

        switch (inclusion) {
            case before : os.append("B "); break;
            case after  : os.append("A "); break;
            default        : throw 4;
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
        for (int i = 1; i < 10; i++) {
            MyWeakHypothesis *h = new MyWeakHypothesis(MyWeakHypothesis::vertical, MyWeakHypothesis::before, i);
            hypothesis.insert(hypothesis.end(), h);

            h = new MyWeakHypothesis(MyWeakHypothesis::vertical, MyWeakHypothesis::after, i);
            hypothesis.insert(hypothesis.end(), h);

            h = new MyWeakHypothesis(MyWeakHypothesis::horizontal, MyWeakHypothesis::before, i);
            hypothesis.insert(hypothesis.end(), h);

            h = new MyWeakHypothesis(MyWeakHypothesis::horizontal, MyWeakHypothesis::after, i);
            hypothesis.insert(hypothesis.end(), h);
        }
    }

    ~MyWeakLearner() {
        for (std::vector < MyWeakHypothesis* >::const_iterator it = hypothesis.begin(); it != hypothesis.end(); it++) {
            delete *it;
        }
        hypothesis.clear();
    }

    virtual WeakHypothesis<Point>* learn(
            const std::vector < LabeledExample<Point> * > &training_set,
            const std::vector < weight_type > &weight_distribution,
            weight_type &weighted_error) {

        weight_type lowest_error = std::numeric_limits<weight_type>::max();
        MyWeakHypothesis* best_hypothesis = 0;

        for (std::vector < MyWeakHypothesis* >::const_iterator it = hypothesis.begin(); it != hypothesis.end(); it++) {
            MyWeakHypothesis const * const hyp = *it;

            weight_type hypothesis_weighted_error = 0;

            for (std::vector < LabeledExample<Point> * >::size_type j = 0; j < training_set.size(); j++) {
                if (hyp->classify(training_set[j]->example) != training_set[j]->label) {
                    hypothesis_weighted_error += weight_distribution[j];
                }
            }

            if (hypothesis_weighted_error < lowest_error) {
                lowest_error = hypothesis_weighted_error;
                best_hypothesis = *it;
            }
        }

        const weight_type maximum_weighted_error = 0.5;
        if (lowest_error < maximum_weighted_error //checks for the weak learning assumption
                || best_hypothesis == 0) { //checks if some hypothesis was selected
            throw 10;
        }

        weighted_error = lowest_error;

        return best_hypothesis;
    }
};



void get_training_data(std::vector < LabeledExample <Point> * > & data) {
    Point *p;
    LabeledExample<Point> * d;

    p = new Point(3, 5);
    d = new LabeledExample<Point>(*p, yes);
    data.push_back(d);

    p = new Point(4, 4);
    d = new LabeledExample<Point>(*p, yes);
    data.push_back(d);

    p = new Point(4, 5);
    d = new LabeledExample<Point>(*p, yes);
    data.push_back(d);

    p = new Point(5, 4);
    d = new LabeledExample<Point>(*p, yes);
    data.push_back(d);

    p = new Point(5, 5);
    d = new LabeledExample<Point>(*p, yes);
    data.push_back(d);

    p = new Point(6, 5);
    d = new LabeledExample<Point>(*p, yes);
    data.push_back(d);

    p = new Point(6, 6);
    d = new LabeledExample<Point>(*p, yes);
    data.push_back(d);

    p = new Point(7, 4);
    d = new LabeledExample<Point>(*p, yes);
    data.push_back(d);

    p = new Point(7, 5);
    d = new LabeledExample<Point>(*p, yes);
    data.push_back(d);

    p = new Point(8, 4);
    d = new LabeledExample<Point>(*p, yes);
    data.push_back(d);

    p = new Point(1, 1);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(1, 2);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(1, 8);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(1, 9);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(2, 1);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(2, 2);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(2, 3);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(2, 4);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(2, 5);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(2, 6);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(2, 7);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(2, 8);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(2, 9);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(3, 1);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(3, 2);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(3, 3);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(3, 4);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(3, 6);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(3, 7);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(3, 8);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(4, 1);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(4, 2);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(4, 3);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(4, 6);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(4, 7);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(4, 8);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(4, 9);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(5, 1);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(5, 2);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(5, 3);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(5, 6);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(5, 7);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(5, 8);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(5, 9);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(6, 1);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(6, 2);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(6, 3);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(6, 4);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(6, 7);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(6, 8);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(6, 9);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(7, 1);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(7, 2);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(7, 3);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(7, 6);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(7, 7);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(8, 1);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(8, 2);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(8, 3);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(8, 5);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(8, 6);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(8, 7);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(9, 1);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(9, 2);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(9, 3);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(9, 4);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(9, 5);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(9, 6);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);

    p = new Point(9, 7);
    d = new LabeledExample<Point>(*p, no);
    data.push_back(d);
}



int main(int argc, char **argv) {
    if (argc != 2)
    {
        return -1;
    }
    const unsigned int maximum_iterations = strtol(argv[1], 0, 10);

    std::vector < LabeledExample <Point> * > training_set;
    get_training_data(training_set);

    WeakLearner<Point> *learner = new MyWeakLearner();
    StrongHypothesis<Point> strong_hypothesis;
    Adaboost<Point> boosting(learner);

    try {
        boosting.train(training_set, strong_hypothesis, maximum_iterations);
    } catch (int e) {
        std::cout << "Erro durante a execução do treinamento. Número do erro: " << e << std::endl;
    }

    int false_positive = 0;
    int true_negative = 0; //positive samples that were classified as negative.

    for (std::vector < LabeledExample <Point> * >::const_iterator it = training_set.begin(); it != training_set.end(); ++it) {
        LabeledExample<Point> * const le = *it;
        const Classification c = strong_hypothesis.classify(le->example);
        if (le->label == yes) {
            if (c == no) {
                true_negative++;
            }
        } else /* le->label == no*/ {
            if (c == yes) {
                false_positive++;
            }
        }

        std::cout << le->example << " esperado " << le->label << " obtido " << c << std::endl;
    }

    std::cout << std::endl << strong_hypothesis << std::endl;
    std::cout << "True negative        : " << true_negative  / (double)training_set.size() << std::endl;
    std::cout << "False positive       : " << false_positive / (double)training_set.size() << std::endl;
    std::cout << "Misclassified        : " << (true_negative + false_positive) / (double)training_set.size() << std::endl;
    std::cout << "Correctly classified : " << (training_set.size() - true_negative - false_positive) / (double)training_set.size() << std::endl;

    return 0;
}

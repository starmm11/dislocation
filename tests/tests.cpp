//
// Created by home on 17.06.17.
//

#include <memory>
#include <algorithm>
#include <gtest/gtest.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/vector.h>


#include "elastic.hpp"
#include "dislocation.hpp"
#include "consts.h"

using namespace dealii;
using namespace std;

struct VectorRotateData {
    double xs;
    double ys;
    double angle;
    double xr;
    double yr;
};

class TestTranspose : public ::testing::TestWithParam<VectorRotateData> {
};


TEST(TestEdgeStress, SimpleTest) {
    vector<EdgeDislocation<2> > disl;
    Elastic elastic(/*lambda*/ 60*1e9, /*mu*/ 27*1e9);
    disl.push_back(EdgeDislocation<2>(Point<2>(0, 1e-5), Vector<double>(2), POSITIVE, 0.0));
    disl.push_back(EdgeDislocation<2>(Point<2>(0, -1e-5), Vector<double>(2), NEGATIVE, 0.0));
    Point<2> p = Point<2>(0, 0);
    SymmetricTensor<2,2> str1 = disl[0].getStress(p, elastic);
    SymmetricTensor<2,2> str2 = disl[1].getStress(p, elastic);
    ASSERT_FLOAT_EQ(str1[1][1], -str2[1][1]);
    ASSERT_FLOAT_EQ(str1[0][0], -str2[0][0]);
}

class Generator {
public:
    Generator(double a, double b) : a(a), b(b) {}
    VectorRotateData operator()() {
        VectorRotateData data;
        data.xs = a + (double)rand()/RAND_MAX * (b-a);
        data.ys = a + (double)rand()/RAND_MAX * (b-a);
        data.angle = (double)rand()/RAND_MAX * 360.0;
        double angle_rad = data.angle*PI_180;
        data.xr = data.xs*cos(angle_rad) + data.ys*sin(angle_rad);
        data.yr = -data.xs*sin(angle_rad) + data.ys*cos(angle_rad);
        return data;
    }
private:
    double a;
    double b;

};

vector<VectorRotateData> createData(int n) {
    Generator generator(0.0, 1.0);
    vector<VectorRotateData> v(n);
    generate(v.begin(), v.end(), generator);
    return v;
}


INSTANTIATE_TEST_CASE_P(TestWithParameters, TestTranspose,
                        ::testing::Values(VectorRotateData{1, 0, 90.0, 0, -1}));

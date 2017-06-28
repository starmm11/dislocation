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

TEST_P(TestTranspose, SimpleTest) {
    const VectorRotateData& p = GetParam();
    Tensor<1, 2> check;
    check[0] = p.xs; check[1] = p.ys;
    Tensor<2, 2> rotation = get_rotation_matrix(p.angle*PI_180);
    Tensor<1, 2> res = check*transpose(rotation);
    EXPECT_NEAR(p.xr, res[0], ERROR);
    EXPECT_NEAR(p.yr, res[1], ERROR);
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

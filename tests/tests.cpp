//
// Created by home on 17.06.17.
//
#include <gtest/gtest.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/vector.h>
#include <memory>

using namespace dealii;
using namespace std;

template<int dim>
class DivideFunction : public Function<dim> {
public:
    void vector_value (const Point<dim> &p, Vector<double> &values) const {
        for (int i = 0; i < values.size(); ++i) {
            values[i] = p[i];
        }
    }
};

TEST(Test, SimpleTest) {
    Vector<double> check(2);
    shared_ptr<Function<2> > function = make_shared<DivideFunction<2> >();
    function->vector_value({1,1}, check);
    EXPECT_EQ(1, check[0]);
    EXPECT_EQ(1, check[1]);
}


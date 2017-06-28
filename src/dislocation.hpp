#ifndef DISLOCATION_HPP_
#define DISLOCATION_HPP_

#include <deal.II/base/point.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/lac/vector.h>
#include "consts.h"
#include "elastic.hpp"

using namespace dealii;

enum Sign {NEGATIVE = -1, POSITIVE = 1};

//class for single dislocation
template<int dim>
class EdgeDislocation {
public:
    EdgeDislocation() {
        coord = Point<dim>();
        v = Vector<double>();
        f = Vector<double>();
        sign = POSITIVE;
        angle = 0;

    }

    EdgeDislocation(const Point<dim>& p, const Vector<double>& v, Sign sign, double angle)
            : coord(p), v(v), sign(sign), angle(angle) {}

    SymmetricTensor<2,dim>
    getStress(const Point<dim> &p,
              const Elastic &e);

    Tensor<1,dim>
    getU(const Point<dim> &p,
         const Elastic &e);

private:
    Point<dim> coord;
    Vector<double> v;
    Vector<double> f;
    Sign sign;
    double angle;


};

#include "dislocation.cpp"

#endif //DISLOCATION_HPP

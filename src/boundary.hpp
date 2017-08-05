/*
 * boundary.hpp
 *
 *  Created on: Aug 30, 2016
 *      Author: home
 */

#ifndef BOUNDARY_HPP_
#define BOUNDARY_HPP_

#include <vector>
#include <utility>

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include "elastic.hpp"
#include "ElasticProblem.hpp"
#include "dislocation.hpp"


using namespace dealii;

template<int dim>
class ElasticProblem;

template <int dim>
class BoundaryValuesU :  public Function<dim> {

public:
    BoundaryValuesU(ElasticProblem<dim>* p);
    virtual void
    vector_value (const Point<dim> &p, Vector<double> &values) const;

    virtual void
    vector_value_list (const std::vector<Point<dim> > &points,
                       std::vector<Vector<double> > &value_list) const;

private:
    ElasticProblem<dim>* ep;

};

template <int dim>
class BoundaryValuesForce : public Function<dim>
{
public:

    BoundaryValuesForce(ElasticProblem<dim>* p);
    void force_value(const Point<dim> &p,
        			Tensor<1,dim>& value,
        			Tensor<1,dim> normal) const;

private:
    ElasticProblem<dim>* ep;
};



#include "boundary.cpp"

#endif /* BOUNDARY_HPP_ */

/*
 * boundary.hpp
 *
 *  Created on: Aug 30, 2016
 *      Author: home
 */

#ifndef BOUNDARY_HPP_
#define BOUNDARY_HPP_

#include <vector>

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include "elastic.hpp"
#include "dislocation.hpp"


using namespace dealii;



template <int dim>
class BoundaryValuesU :  public Function<dim> {

public:
    BoundaryValuesU(const Tensor<1,dim>& boundary_displacement,
                    Elastic* elastic,
                    std::vector<std::vector<EdgeDislocation<dim> > >* disl,
                    std::vector<double>* angles,
                    std::vector<Tensor<2,2> >* rotation_matrices
                    );

    virtual void
    vector_value (const Point<dim> &p, Vector<double> &values) const;

    virtual void
    vector_value_list (const std::vector<Point<dim> > &points,
                       std::vector<Vector<double> > &value_list) const;

private:
    Tensor<1,dim> displacement;
    std::vector<std::vector<EdgeDislocation<dim> > >* dislocations;
    Elastic* elastic;
    std::vector<Tensor<2,2> >* rotation_matrices;
    std::vector<double>* angles;
    int number_of_slip_systems;
};

template <int dim>
class BoundaryValuesForce : public Function<dim>
{
public:

    BoundaryValuesForce(const Tensor<1,dim>& boundary_force,
                        Elastic* elastic,
                        std::vector<std::vector<EdgeDislocation<dim> > >* disl,
                        std::vector<double>* angles,
                        std::vector<Tensor<2,2> >* rotation_matrices);

    void force_value(const Point<dim> &p,
        			Tensor<1,dim>& value,
        			Tensor<1,dim> normal) const;

private:
    Tensor<1,dim> boundary_force;
    std::vector<std::vector<EdgeDislocation <dim> > >* dislocations;
    Elastic* elastic;
    std::vector<Tensor<2,2> >* rotation_matrices;
    std::vector<double>* angles;
    int number_of_slip_systems;
};

#include "boundary.cpp"

#endif /* BOUNDARY_HPP_ */

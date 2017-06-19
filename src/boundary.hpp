/*
 * boundary.hpp
 *
 *  Created on: Aug 30, 2016
 *      Author: home
 */

#ifndef BOUNDARY_HPP_
#define BOUNDARY_HPP_

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

using namespace dealii;

template <int dim>
class BoundaryValuesU :  public Function<dim>
{
   public:
	 BoundaryValuesU(EdgeDislocation<dim>* d, Tensor<1,dim> displacement,
			         double lambda, double mu, double burgers);
	 //overload
     void vector_value (const Point<dim> &p, Vector<double> &values) const;

     //overload
     void vector_value_list (const std::vector<Point<dim> > &points,
                        std::vector<Vector<double> >   &value_list) const;

   private:
     EdgeDislocation<dim>* disl;
     Tensor<1,dim>  displacement;
     double lambda;
     double mu;
     double b;
};

template <int dim>
class BoundaryValuesForce : public Function<dim>
{
	public:
		BoundaryValuesForce(EdgeDislocation<dim>* d,
				const Tensor<1,dim>& boundary_force,
		        double lambda, double mu, double burgers);

        void force_value(const Point<dim> &p,
        			Tensor<1,dim>& value,
        			Tensor<1,dim> normal) const;

	private:
        EdgeDislocation<dim>* disl;
        Tensor<1,dim>  force;
        double lambda;
        double mu;
        double b;
};

#include "boundary.cpp"

#endif /* BOUNDARY_HPP_ */

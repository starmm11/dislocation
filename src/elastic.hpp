/*
 * elastic.hpp
 *
 *  Created on: Aug 30, 2016
 *      Author: home
 */

#ifndef ELASTIC_HPP_
#define ELASTIC_HPP_

#include <deal.II/base/point.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/fe/fe_values.h>
#include <cmath>

using namespace dealii;

struct Elastic {
    Elastic() : lambda(0), mu(0), poisson(0), shear(0), young(0) {}
    Elastic(double lambda, double mu) : lambda(lambda), mu(mu) {
        poisson = 0.5 * lambda / (lambda + mu);
        shear = mu;
        young = mu*(3*lambda+2*mu)/(lambda+mu);
    }
    double lambda;
    double mu;
    double poisson;
    double shear;
    double young;
};

// Clockwise rotation matrix
Tensor<2,2>
get_rotation_matrix(double angle) {
	const double t[2][2] = {{std::cos(angle), std::sin(angle)},
			{-std::sin(angle), std::cos(angle)}};
	return Tensor<2,2>(t);
}


template <int dim>
SymmetricTensor<4,dim>
get_stress_strain_tensor (double lambda, double mu)
{
	SymmetricTensor<4,dim> tmp;
    for (unsigned int i=0; i<dim; ++i)
    	for (unsigned int j=0; j<dim; ++j)
    		for (unsigned int k=0; k<dim; ++k)
    			for (unsigned int l=0; l<dim; ++l)
    				tmp[i][j][k][l] = (((i==k) && (j==l) ? mu : 0.0) +
                                   ((i==l) && (j==k) ? mu : 0.0) +
                                   ((i==j) && (k==l) ? lambda : 0.0));
     return tmp;
}

// strain tensor
template <int dim>
inline
SymmetricTensor<2,dim>
get_strain (const FEValues<dim> &fe_values,
            const unsigned int   shape_func,
            const unsigned int   q_point)
{
      // Declare a temporary that will hold the return value:
    SymmetricTensor<2,dim> tmp;

    for (unsigned int i=0; i<dim; ++i)
        tmp[i][i] = fe_values.shape_grad_component (shape_func,q_point,i)[i];

    for (unsigned int i=0; i<dim; ++i)
        for (unsigned int j=i+1; j<dim; ++j)
          tmp[i][j]
            = (fe_values.shape_grad_component (shape_func,q_point,i)[j] +
               fe_values.shape_grad_component (shape_func,q_point,j)[i]) / 2;

    return tmp;
}


// strain tensor
template <int dim>
inline
SymmetricTensor<2,dim>
get_strain (const std::vector<Tensor<1,dim> > &grad) {
    Assert (grad.size() == dim, ExcInternalError());

    SymmetricTensor<2,dim> strain;
    for (unsigned int i=0; i<dim; ++i)
        strain[i][i] = grad[i][i];

    for (unsigned int i=0; i<dim; ++i)
        for (unsigned int j=i+1; j<dim; ++j)
          strain[i][j] = (grad[i][j] + grad[j][i]) / 2;

    return strain;
}

#endif /* ELASTIC_HPP_ */

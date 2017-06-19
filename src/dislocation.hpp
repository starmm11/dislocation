#ifndef DISLOCATION_HPP_
#define DISLOCATION_HPP_

#include <deal.II/base/point.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/lac/vector.h>
#include "consts.h"

using namespace dealii;


//class for single dislocation
template<int dim>
class EdgeDislocation {
	public:
		EdgeDislocation() { coord = Point<dim>(); }
		EdgeDislocation(const Point<dim>& p) : coord(p) {}

	    SymmetricTensor<2,dim>
	    get_stress(const Point<dim>& p,
	    		   const double& lambda, const double& mu,
	    		   const double& b
	    		   );

	    Tensor<1,dim>
	    get_u(const Point<dim>& p,
	    	  const double& lambda, const double& mu,
	    	  const double& b);
	private:
	    Point<dim> coord;
};

#include "dislocation.cpp"

#endif //DISLOCATION_HPP

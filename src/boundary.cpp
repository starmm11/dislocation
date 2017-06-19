#include "boundary.hpp"

template <int dim>
BoundaryValuesU<dim>::BoundaryValuesU(EdgeDislocation<dim>* d, Tensor<1,dim> displacement,
                                 double lambda, double mu, double burgers) :
    Function<dim> (dim),
    disl(d),
    displacement(displacement),
    lambda(lambda),
    mu(mu),
    b(burgers) {}

template <int dim>
void
BoundaryValuesU<dim>::vector_value (const Point<dim> &p,
                               Vector<double>   &values) const
{
    Assert (values.size() == dim, ExcDimensionMismatch (values.size(), dim));
    const Tensor<1,dim> dislocation_u = disl->get_u(p,lambda, mu, b);
    values(0) = displacement[0] - dislocation_u[0];
    values(1) = displacement[1] - dislocation_u[1];
}

template <int dim>
void
BoundaryValuesU<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                    std::vector<Vector<double> >   &value_list) const
{
    const unsigned int n_points = points.size();

    Assert (value_list.size() == n_points,
            ExcDimensionMismatch (value_list.size(), n_points));

    for (unsigned int p=0; p<n_points; ++p)
        vector_value (points[p], value_list[p]);
}

template <int dim>
BoundaryValuesForce<dim>::BoundaryValuesForce(EdgeDislocation<dim>* d,
				    const Tensor<1,dim>& boundary_force,
		            double lambda, double mu, double burgers) :
			disl(d),
			force(boundary_force),
			lambda(lambda),
			mu(mu),
			b(burgers) {}

template <int dim>
void
BoundaryValuesForce<dim>::force_value(const Point<dim> &p,
                                 Tensor<1, dim>& value,
                                 Tensor<1,dim> normal) const
{
    Tensor<1, dim> force_from_dislocation = disl->get_stress(p,lambda, mu, b) * normal;

    value[0] = force[0] - force_from_dislocation[0];
    value[1] = force[1] - force_from_dislocation[1];
}




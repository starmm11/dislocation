#include "boundary.hpp"

template <int dim>
BoundaryValuesU<dim>::BoundaryValuesU(ElasticProblem<dim>* p) :
    Function<dim> (dim), ep(p)
{}


template <int dim>
void
BoundaryValuesU<dim>::vector_value (const Point<dim> &p,
                                    Vector<double>   &values) const
{
    Assert (values.size() == dim, ExcDimensionMismatch (values.size(), dim));
    std::vector<Tensor<1,dim> > dislocation_u(ep->number_of_slip_systems);
    Tensor<1,dim> overall_dislocation_u;
    // Compute the contribution of each slip class of dislocations to
    // dislocation_u[k]
    for (int k = 0; k < ep->number_of_slip_systems; ++k) {
        for (size_t i = 0; i < ep->dislocations[k].size(); ++i) {
            dislocation_u[k] += ep->dislocations[k][i].getU(p, ep->elastic);
        }
    }

    // Gather all displacement into overall_dislocation_u
    // by rotating clockwise contributions of each slip class of dislocation
    // using rotation_matrices[k]

    for (int k = 0; k < ep->number_of_slip_systems; ++k) {
        if (abs(ep->slip_system_angles[k] - 0.0) < ERROR) {
            overall_dislocation_u += dislocation_u[k];
        }
        else {
            // to rotate vector we need to
            // (xnorm, ynorm) = (x, y) * ({cos(a), sin(a)}, {-sin(a), cos(a)})
            overall_dislocation_u += dislocation_u[k]*transpose(ep->rotation_matrices[k]);
        }
    }

    values(0) = ep->boundary_displacement[0] - overall_dislocation_u[0];
    values(1) = ep->boundary_displacement[1] - overall_dislocation_u[1];
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
BoundaryValuesForce<dim>::
BoundaryValuesForce(ElasticProblem<dim>* p) :
        Function<dim> (dim), ep(p)
{

}


template <int dim>
void
BoundaryValuesForce<dim>::force_value(const Point<dim> &p,
                                 Tensor<1, dim>& value,
                                 Tensor<1,dim> normal) const
{
    // create temporary data
    std::vector<SymmetricTensor<2,dim> > stress_from_dislocations(ep->number_of_slip_systems);
    SymmetricTensor<2,dim> overall_stress_from_dislocations;
    Tensor<1, dim> force_from_dislocations;
    // Compute the contribution of each slip class of dislocations to
    // stress_from_dislocations[k]
    for (int k = 0; k < ep->number_of_slip_systems; ++k) {
        for (size_t i = 0; i < ep->dislocations[k].size(); ++i) {
            stress_from_dislocations[k] += ep->dislocations[k][i].getStress(p, ep->elastic);
        }
    }
    // Gather all displacement into overall_stress_from_dislocations
    // by rotating clockwise contributions of each slip class of dislocation
    // using rotation_matrices[k]

    for (int k = 0; k < ep->number_of_slip_systems; ++k) {
        if (abs(ep->slip_system_angles[k] - 0.0) < ERROR) {
            overall_stress_from_dislocations += stress_from_dislocations[k];
        }
        else {
            overall_stress_from_dislocations +=
                    symmetrize(ep->rotation_matrices[k]*
                    static_cast<Tensor<2, dim> >(stress_from_dislocations[k]) *
                    transpose(ep->rotation_matrices[k]));
        }
    }

    force_from_dislocations = overall_stress_from_dislocations * normal;

    // save value to vector
    ep->force_at_boundary_solution.push_back(std::make_pair(p, force_from_dislocations));

    value[0] = ep->boundary_force[0] - force_from_dislocations[0];
    value[1] = ep->boundary_force[1] - force_from_dislocations[1];
}




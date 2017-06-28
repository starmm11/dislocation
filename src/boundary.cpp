#include "boundary.hpp"

template <int dim>
BoundaryValuesU<dim>::BoundaryValuesU(const Tensor<1,dim>& boundary_displacement,
                                      Elastic* elastic,
                                      std::vector<std::vector<EdgeDislocation<dim> > >* disl,
                                      std::vector<double>* angles,
                                      std::vector<Tensor<2,2> >* rot_m) :
    Function<dim> (dim),
    displacement(boundary_displacement),
    dislocations(disl),
    elastic(elastic),
    rotation_matrices(rot_m),
    angles(angles),
    number_of_slip_systems(disl->size())
    {}

template <int dim>
void
BoundaryValuesU<dim>::vector_value (const Point<dim> &p,
                                    Vector<double>   &values) const
{
    Assert (values.size() == dim, ExcDimensionMismatch (values.size(), dim));

    std::vector<Tensor<1,dim> > dislocation_u(number_of_slip_systems);
    // Compute the contribution of each slip class of dislocations to
    // dislocation_u[k]
    for (int k = 0; k < number_of_slip_systems; ++k) {
        for (size_t i = 0; i < dislocations[k].size(); ++i) {
            dislocation_u[k] += (*dislocations)[k][i].getU(p, *elastic);
        }
    }

    // Gather all displacement into overall_dislocation_u
    // by rotating clockwise contributions of each slip class of dislocation
    // using rotation_matrices[k]
    Tensor<1,dim> overall_dislocation_u;
    for (int k = 0; k < number_of_slip_systems; ++k) {
        if (abs((*angles)[k] - 0.0) < ERROR) {
            overall_dislocation_u += dislocation_u[k];
        }
        else {
            // to rotate vector we need to
            // (xnorm, ynorm) = (x, y) * ({cos(a), sin(a)}, {-sin(a), cos(a)})
            overall_dislocation_u += dislocation_u[k]*transpose((*rotation_matrices)[k]);
        }
    }

    values(0) = displacement[0] - overall_dislocation_u[0];
    values(1) = displacement[1] - overall_dislocation_u[1];
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
BoundaryValuesForce(const Tensor<1,dim>& boundary_force,
                    Elastic* elastic,
                    std::vector<std::vector<EdgeDislocation<dim> > >* disl,
                    std::vector<double>* angles,
                    std::vector<Tensor<2,2> >* rot_m) :

        Function<dim> (dim),
        boundary_force(boundary_force),
        dislocations(disl),
        elastic(elastic),
        rotation_matrices(rot_m),
        angles(angles),
        number_of_slip_systems(disl->size())
{}

template <int dim>
void
BoundaryValuesForce<dim>::force_value(const Point<dim> &p,
                                 Tensor<1, dim>& value,
                                 Tensor<1,dim> normal) const
{

    // Compute the contribution of each slip class of dislocations to
    // stress_from_dislocations[k]
    std::vector<SymmetricTensor<2,dim> > stress_from_dislocations(number_of_slip_systems);
    for (int k = 0; k < number_of_slip_systems; ++k) {
        for (size_t i = 0; i < dislocations[k].size(); ++i) {
            stress_from_dislocations[k] += (*dislocations)[k][i].getStress(p, *elastic);
        }
    }
    // Gather all displacement into overall_stress_from_dislocations
    // by rotating clockwise contributions of each slip class of dislocation
    // using rotation_matrices[k]
    SymmetricTensor<2,dim> overall_stress_from_dislocations;
    for (int k = 0; k < number_of_slip_systems; ++k) {
        if (abs((*angles)[k] - 0.0) < ERROR) {
            overall_stress_from_dislocations += stress_from_dislocations[k];
        }
        else {
            overall_stress_from_dislocations +=
                    symmetrize((*rotation_matrices)[k]*
                    static_cast<Tensor<2, dim> >(stress_from_dislocations[k]) *
                    transpose((*rotation_matrices)[k]));
        }
    }

    Tensor<1, dim> force_from_dislocations = overall_stress_from_dislocations * normal;
    value[0] = boundary_force[0] - force_from_dislocations[0];
    value[1] = boundary_force[1] - force_from_dislocations[1];
    std::cout << value[0] << ' ' << value[1] << '\n';
}




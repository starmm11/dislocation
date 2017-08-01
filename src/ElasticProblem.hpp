//
// Created by home on 31.03.17.
//

#ifndef UNIAXIAL_PROBLEM_H
#define UNIAXIAL_PROBLEM_H

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <utility>

#include "dislocation.hpp"
#include "elastic.hpp"
#include "boundary.hpp"
#include "consts.h"


using namespace dealii;

template<int dim>
struct UserData {
    SymmetricTensor<2, dim> full_stress;
    SymmetricTensor<2, dim> dislocation_stress;
    SymmetricTensor<2, dim> image_stress;
    SymmetricTensor<2, dim> strain;
    Tensor<1, dim> quadrature_disl_displacement;
    Tensor<1, dim> force;
};

template<int dim>
class ElasticProblem {
public:
    ElasticProblem();
    ~ElasticProblem();
    void Run();

    friend class BoundaryValuesForce<dim>;
    friend class BoundaryValuesU<dim>;
private:
    void CreateGrid(double size, int n_refine);
    void CreateOneDislocation(const Point<dim> &p, const Vector<double> &v,
                              Sign sign, double angle);
    void CreateDislocations(const std::vector<Point<dim> >& points,
                            const std::vector<Sign>& signs,
                            const std::vector<double>& angles);
    void SetupSystem();
    void SetupSlipSystems(const std::vector<double>& angles);

    // Need to be updated
    void SetupBoundaryConditions(const Tensor<1, dim>& force,
                                 const Tensor<1, dim>& displacement);
    void AddDislocationComponentToResult();
    void SetupQuadratureUserData();
    void UpdateUserData();
    void SetupElasticProperties(double lambda, double mu);
    void AssembleSystem();
    void AssembleSystemTensor();
    void Solve();
    void RefineGrid();
    void OutputResults(const unsigned int cycle) const;
    void ComputeBoundaryForceSolution();

    //Dislocation motion functions
    //Waiting for realization!!
    void ComputeStresses();
    void ComputeDisplacement();
    void ComputeForces();
    void MoveDislocations();

private:
    Triangulation<dim> triangulation;
    DoFHandler<dim> dof_handler;
    FESystem<dim> fe;
    ConstraintMatrix hanging_node_constraints;
    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    QGauss<dim> quadrature_formula;
    QGauss<dim-1> face_quadrature_formula;

    // Boundary class
    BoundaryValuesForce<dim> boundary_force_condition;
    Tensor<1, dim> boundary_force;
    BoundaryValuesU<dim> boundary_u_condition;
    Tensor<1, dim> boundary_displacement;

    // Solution data
    Vector<double> image_solution;
    Vector<double> dislocation_solution;
    Vector<double> overall_solution;
    Vector<double> system_rhs;

    std::vector<std::pair<Point<dim> , Tensor<1, dim> > > force_at_boundary_solution;

    // Quadrature Data
    std::vector<UserData<dim> > quadrature_user_data;
    std::vector<SymmetricTensor<2, dim> > image_stress_cell;

    // Elastic properties
    SymmetricTensor<4, dim> stress_strain_tensor;
    Elastic elastic;

    // Array of dislocations for each angle
    std::vector<std::vector<EdgeDislocation<dim> > > dislocations;

    // Number of slip systems
    int number_of_slip_systems;

    // Angles of slip systems
    std::vector<double> slip_system_angles;

    // Number of dislocations in each slip system
    std::vector<int> slip_system_sizes;

    // Rotation matrix for each slip system
    // needs to be refactoring
    std::vector<Tensor<2,2> > rotation_matrices;

    // Dislocation motion integration properties
    // Timestep
    double timestep;




};


#include "ElasticProblem.cpp"



#endif //UNIAXIAL_PROBLEM_H

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

#include "dislocation.hpp"
#include "elastic.hpp"
#include "boundary.hpp"


namespace DDisl {

    using namespace dealii;

    template<int dim>
    struct UserData {
        SymmetricTensor<2, dim> quadrature_stress;
        SymmetricTensor<2, dim> quadrature_strain;
        Tensor<1, dim> quadrature_disl_displ;
    };

    template<int dim>
    class ElasticProblem {
    public:
        ElasticProblem();
        ~ElasticProblem();
        void run();

    private:
        void create_grid(double size, int n_refine);
        void create_dislocation(const Point<dim> &p);

        //Waiting for realization!!
        void create_dislocation_system(
                int n_dislocations,
                std::vector<double> angles,
                std::vector<double> slip_systems_spaces,
                int n_slip_systems);

        void setup_system();
        void add_dislocation_component_to_result();
        void setup_quadrature_user_data();
        void update_user_data();
        void setup_elastic(double lambda, double mu, double b);
        void assemble_system();
        void assemble_system_tensor();
        void solve();
        void refine_grid();
        void output_results(const unsigned int cycle) const;

        //Dislocation motion functions
        //Waiting for realization!!
        void compute_forces();
        void move_dislocations();

    private:
        Triangulation<dim> triangulation;
        DoFHandler<dim> dof_handler;
        FESystem<dim> fe;
        ConstraintMatrix hanging_node_constraints;
        SparsityPattern sparsity_pattern;
        SparseMatrix<double> system_matrix;
        QGauss<dim> quadrature_formula;
        QGauss<dim - 1> face_quadrature_formula;

        Vector<double> image_solution;
        Vector<double> dislocation_solution;
        Vector<double> overall_solution;
        Vector<double> system_rhs;
        std::vector<UserData<dim> > quadrature_user_data;
        std::vector<SymmetricTensor<2, dim> > image_stress_cell;

        EdgeDislocation <dim> *dislocation;
        std::vector<EdgeDislocation < dim> > dislocations;

        // Elastic properties
        SymmetricTensor<4, dim> stress_strain_tensor;
        double lambda;
        double mu;
        double burgers;

        // Dislocation motion integration properties
        double timestep;
    };

    #include "ElasticProblem.cpp"
}


#endif //UNIAXIAL_PROBLEM_H

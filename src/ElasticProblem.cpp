//
// Created by home on 31.03.17.
//

#ifndef ELASTIC_PROBLEM_H
#define ELASTIC_PROBLEM_H

#include "ElasticProblem.hpp"

// @sect4{ElasticProblem::ElasticProblem}
template <int dim>
ElasticProblem<dim>::ElasticProblem ()
        :
        dof_handler (triangulation),
        fe (FE_Q<dim>(1), dim),
        quadrature_formula(2),
        face_quadrature_formula(2),
        boundary_force_condition(this),
        boundary_u_condition(this)
{}

template <int dim>
ElasticProblem<dim>::~ElasticProblem () {
    dof_handler.clear();
}


// @sect4{ElasticProblem::setup_system}
template <int dim>
void ElasticProblem<dim>::CreateGrid(double size, int n_refine) {
    GridGenerator::hyper_cube(triangulation, -size, size);
    std::cout << "Cube of [" << -size << ", " << size << "] is generated\n";
    std::cout << "Number of active cells: " <<
              triangulation.n_active_cells() << std::endl;

    /*for (typename Triangulation<dim>::active_cell_iterator
         cell=triangulation.begin_active();
         cell!=triangulation.end(); ++cell)
      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
        if (cell->face(f)->at_boundary()) {
           const Point<dim> face_center = cell->face(f)->center();
           if (face_center[0] == -size)
               cell->face(f)->set_boundary_id (1);
           else if (face_center[0] == size)
               cell->face(f)->set_boundary_id (2);
           else
               cell->face(f)->set_boundary_id (0);

        }*/
    triangulation.refine_global (n_refine);
}

template <int dim>
void ElasticProblem<dim>::CreateOneDislocation(const Point<dim> &p, const Vector<double> &v,
                                               Sign sign, double angle) {
    for (int k = 0; k < number_of_slip_systems; ++k) {
        if (angle == slip_system_angles[k]) {
            dislocations[k][slip_system_sizes[k]++] =
                    EdgeDislocation<dim>(p, v, sign, angle);
        }
    }
}

template <int dim>
void ElasticProblem<dim>::CreateDislocations(const std::vector<Point<dim> >& points,
                                             const std::vector<Sign>& signs,
                                             const std::vector<double>& angles)
{
    dislocations.resize(number_of_slip_systems);
    slip_system_sizes.resize(number_of_slip_systems);
    //resize dislocations arrays by number of given data
    // needs to be refactoring...
    for (int k = 0; k < number_of_slip_systems; ++k) {
        dislocations[k].resize(points.size());
    }

    for (int i = 0; i < points.size(); ++i) {
        for (int k = 0; k < number_of_slip_systems; ++k) {
            if (angles[i] == slip_system_angles[k]) {
                dislocations[k][slip_system_sizes[k]++] =
                        EdgeDislocation<dim>(points[i], Vector<double>(dim),
                                             signs[i], angles[i]);
            }
        }
    }
}

template <int dim>
void ElasticProblem<dim>::SetupSlipSystems(const std::vector<double>& angles) {
    number_of_slip_systems = angles.size();
    slip_system_angles = angles;
    rotation_matrices.reserve(number_of_slip_systems);
    for (int i = 0; i < number_of_slip_systems; ++i) {
        rotation_matrices[i] = get_rotation_matrix(angles[i]);
    }
}

template <int dim>
void ElasticProblem<dim>::SetupSystem() {
    dof_handler.distribute_dofs (fe);
    hanging_node_constraints.clear ();
    DoFTools::make_hanging_node_constraints (dof_handler,
                                             hanging_node_constraints);
    hanging_node_constraints.close ();

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    hanging_node_constraints,
            /*keep_constrained_dofs = */ true);
    sparsity_pattern.copy_from (dsp);

    system_matrix.reinit (sparsity_pattern);
    std::cout << "Dof_handler number: " <<
              dof_handler.n_dofs() << std::endl;
    image_solution.reinit (dof_handler.n_dofs());
    dislocation_solution.reinit (dof_handler.n_dofs());
    overall_solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
}

template <int dim>
void ElasticProblem<dim>::SetupElasticProperties(double lambda, double mu) {
    elastic = Elastic(lambda, mu);
    stress_strain_tensor = get_stress_strain_tensor<dim>(lambda, mu);
}

template <int dim>
void ElasticProblem<dim>::SetupBoundaryConditions(const Tensor<1, dim>& force,
                                             const Tensor<1, dim>& displacement) {
    boundary_force = force;
    boundary_displacement = displacement;
}
// @sect4{ElasticProblem::assemble_system}

template <int dim>
void ElasticProblem<dim>::AssembleSystemTensor() {

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values   | update_gradients |
                             update_quadrature_points | update_JxW_values);

    FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                      update_values   | update_normal_vectors |
                                      update_quadrature_points | update_JxW_values);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();
    const unsigned int   n_face_q_points = face_quadrature_formula.size();

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);


    std::vector<Vector<double> > rhs_values (n_q_points,
                                             Vector<double>(dim));

    // Now we can begin with the loop over all cells:
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
            endc = dof_handler.end();
    std::cout << "Assembling right hand side\n";
    for (; cell!=endc; ++cell) {
        cell_matrix = 0;
        cell_rhs = 0;

        fe_values.reinit (cell);

        // assembling
        for (unsigned int i=0; i<dofs_per_cell; ++i) {
            for (unsigned int j=0; j<dofs_per_cell; ++j) {
                for (unsigned int q_point=0; q_point<n_q_points;
                     ++q_point) {
                    const SymmetricTensor<2, dim>
                            eps_phi_i = get_strain(fe_values, i, q_point),
                            eps_phi_j = get_strain(fe_values, j, q_point);
                    cell_matrix(i,j) +=
                            (eps_phi_i * stress_strain_tensor * eps_phi_j
                             *
                             fe_values.JxW(q_point));
                }
            }
        }


        // Assembling the right hand side boundary terms
        for (unsigned int face_number=0;
             face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
        {
            Tensor<1, dim> neumann_vector_value;
            if (cell->face(face_number)->at_boundary() &&
                cell->face(face_number)->boundary_id() == 0)
            {
                fe_face_values.reinit(cell, face_number);
                // quadrature
                for(unsigned int q = 0; q < n_face_q_points; ++q)
                {
                    boundary_force_condition.force_value(
                            fe_face_values.quadrature_point(q),
                            neumann_vector_value,
                            fe_face_values.normal_vector(q)
                    );

                    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                        const unsigned int component_i = fe.system_to_component_index(i).first;
                        cell_rhs(i) += neumann_vector_value[component_i] *
                                       fe_face_values.shape_value(i, q) *
                                       fe_face_values.JxW(q);
                    }
                }
            }
        }
        cell->get_dof_indices (local_dof_indices);
        hanging_node_constraints.
                distribute_local_to_global(cell_matrix, cell_rhs,
                                           local_dof_indices,
                                           system_matrix, system_rhs);
    }

    hanging_node_constraints.condense (system_matrix);
    hanging_node_constraints.condense (system_rhs);

    // boundary
    /*std::map<types::global_dof_index,double> boundary_values;
    Tensor<1,dim> d; d[0] = 0.0; d[1] = 0.0;
    VectorTools::interpolate_boundary_values (dof_handler,
                                              1,
                                              boundary_u_condition;
    VectorTools::interpolate_boundary_values (dof_handler,
                                              2,
                                              boundary_u_condition);
    MatrixTools::apply_boundary_values (boundary_values,
                                        system_matrix,
                                        solution,
                                        system_rhs);*/
}

template <int dim>
void ElasticProblem<dim>::AssembleSystem() {
    QGauss<dim>  quadrature_formula(3);
    QGauss<dim-1>  face_quadrature_formula(3);

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values   | update_gradients |
                             update_quadrature_points | update_JxW_values);

    FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                      update_values   | update_normal_vectors |
                                      update_quadrature_points | update_JxW_values);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();
    const unsigned int   n_face_q_points = face_quadrature_formula.size();

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    // right hand side
    std::vector<Vector<double> > rhs_values (n_q_points,
                                             Vector<double>(dim));

    // Now we can begin with the loop over all cells:
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
            endc = dof_handler.end();
    for (; cell!=endc; ++cell) {
        cell_matrix = 0;
        cell_rhs = 0;

        fe_values.reinit (cell);

        // assembling
        for (unsigned int i=0; i<dofs_per_cell; ++i) {
            const unsigned int
                    component_i = fe.system_to_component_index(i).first;
            for (unsigned int j=0; j<dofs_per_cell; ++j) {
                const unsigned int
                        component_j = fe.system_to_component_index(j).first;
                for (unsigned int q_point=0; q_point<n_q_points;
                     ++q_point) {
                    cell_matrix(i,j) +=
                            (
                                    (fe_values.shape_grad(i,q_point)[component_i] *
                                     fe_values.shape_grad(j,q_point)[component_j] *
                                     elastic.lambda)
                                    +
                                    (fe_values.shape_grad(i,q_point)[component_j] *
                                     fe_values.shape_grad(j,q_point)[component_i] *
                                     elastic.mu)
                                    +
                                    ((component_i == component_j) ?
                                     (fe_values.shape_grad(i,q_point) *
                                      fe_values.shape_grad(j,q_point) *
                                      elastic.mu)  :
                                     0)
                            )
                            *
                            fe_values.JxW(q_point);
                }
            }
        }

        // Assembling the right hand side boundary terms
        for (unsigned int face_number=0;
             face_number<GeometryInfo<dim>::faces_per_cell; ++face_number) {
            Tensor<1, dim> neumann_vector_value;
            if (cell->face(face_number)->at_boundary() &&
                cell->face(face_number)->boundary_id() == 0)
            {
                fe_face_values.reinit(cell, face_number);
                // quadrature
                for(unsigned int q = 0; q < n_face_q_points; ++q)
                {

                    boundary_force_condition.force_value(
                            fe_face_values.quadrature_point(q),
                            neumann_vector_value,
                            fe_face_values.normal_vector(q)
                            );
                    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                        const unsigned int
                                component_i = fe.system_to_component_index(i).first;
                        cell_rhs(i) += neumann_vector_value[component_i] *
                                       fe_face_values.shape_value(i, q) *
                                       fe_face_values.JxW(q);
                    }
                }
            }
        }
        cell->get_dof_indices (local_dof_indices);
        for (unsigned int i=0; i<dofs_per_cell; ++i) {
            for (unsigned int j=0; j<dofs_per_cell; ++j) {
                system_matrix.add (local_dof_indices[i],
                                   local_dof_indices[j],
                                   cell_matrix(i,j));
            }
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }

    hanging_node_constraints.condense (system_matrix);
    hanging_node_constraints.condense (system_rhs);

    // boundary
    std::map<types::global_dof_index,double> boundary_values;
    Tensor<1,dim> d; d[0] = 0.0; d[1] = 0.0;
    VectorTools::interpolate_boundary_values (dof_handler,
                                              1,
                                              boundary_u_condition,
                                              boundary_values);
    VectorTools::interpolate_boundary_values (dof_handler,
                                              2,
                                              boundary_u_condition,
                                              boundary_values);
    MatrixTools::apply_boundary_values (boundary_values,
                                        system_matrix,
                                        image_solution,
                                        system_rhs);
}


// @sect4{ElasticProblem::solve}

template <int dim>
void ElasticProblem<dim>::Solve()
{
    SolverControl           solver_control (50000, 1e-6);
    SolverCG<>              cg (solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    cg.solve (system_matrix, image_solution, system_rhs,
              preconditioner);

    hanging_node_constraints.distribute (image_solution);
}

template <int dim>
void ElasticProblem<dim>::AddDislocationComponentToResult() {
    std::vector<bool> vector_touched(triangulation.n_vertices(), false);

    typename DoFHandler<dim>::active_cell_iterator
            cell = dof_handler.begin_active(),
            endc = dof_handler.end();
    for (; cell!=endc; ++cell) {
        for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v) {
            if (vector_touched[cell->vertex_index(v)] == false) {
                // get displacements from dislocations each slip systems
                std::vector<Tensor<1, dim> > u_disl(number_of_slip_systems);
                for (int k = 0; k < number_of_slip_systems; ++k) {
                    for (size_t i = 0; i < dislocations[k].size(); ++i) {
                        u_disl[k] += dislocations[k][i].getU(cell->vertex(v), elastic);
                    }
                }

                for (int k = 0; k < number_of_slip_systems; ++k) {
                    for (unsigned int d = 0; d < dim; ++d) {
                        if (abs(slip_system_angles[k] - 0.0) < ERROR) {
                            dislocation_solution(cell->vertex_dof_index(v, d)) += u_disl[k][d];
                        } else {
                            dislocation_solution(cell->vertex_dof_index(v, d)) +=
                                    (u_disl[k] * transpose(rotation_matrices[k]))[d];
                        }
                    }
                }


                for (unsigned int d = 0; d < dim; ++d) {
                    overall_solution(cell->vertex_dof_index(v,d)) =
                            dislocation_solution(cell->vertex_dof_index(v,d)) +
                            image_solution(cell->vertex_dof_index(v,d));
                }

                vector_touched[cell->vertex_index(v)] = true;
            }
        }
    }
    std::cout << "Vertices: " << vector_touched.size() << '\n';
    std::cout << "Solution: " << overall_solution.size() << '\n';
}
// @sect4{ElasticProblem::refine_grid}

template <int dim>
void ElasticProblem<dim>::RefineGrid()
{
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate (dof_handler,
                                        QGauss<dim-1>(2),
                                        typename FunctionMap<dim>::type(),
                                        overall_solution,
                                        estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                     estimated_error_per_cell,
                                                     0.3, 0.03);

    triangulation.execute_coarsening_and_refinement ();
}


// @sect4{ElasticProblem::output_results}

template <int dim>
void ElasticProblem<dim>::OutputResults(const unsigned int cycle) const
{
    std::string filename = "solution-";
    filename += ('0' + cycle);
    Assert (cycle < 10, ExcInternalError());

    filename += ".vtk";
    std::ofstream output (filename.c_str());

    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);


    std::vector<std::string> solution_names;
    switch (dim)
    {
        case 1:
            solution_names.push_back ("displacement");
            break;
        case 2:
            solution_names.push_back ("x_displacement");
            solution_names.push_back ("y_displacement");
            break;
        case 3:
            solution_names.push_back ("x_displacement");
            solution_names.push_back ("y_displacement");
            solution_names.push_back ("z_displacement");
            break;
        default:
            Assert (false, ExcNotImplemented());
    }

    data_out.add_data_vector (overall_solution, solution_names);
    Vector<double> image_stress11(triangulation.n_active_cells());
    Vector<double> image_stress12(triangulation.n_active_cells());
    Vector<double> image_stress22(triangulation.n_active_cells());
    Vector<double> disl_stress11(triangulation.n_active_cells());
    Vector<double> disl_stress12(triangulation.n_active_cells());
    Vector<double> disl_stress22(triangulation.n_active_cells());
    Vector<double> full_stress11(triangulation.n_active_cells());
    Vector<double> full_stress12(triangulation.n_active_cells());
    Vector<double> full_stress22(triangulation.n_active_cells());
    Vector<double> strain11(triangulation.n_active_cells());
    Vector<double> strain12(triangulation.n_active_cells());
    Vector<double> strain22(triangulation.n_active_cells());

    for (typename Triangulation<dim>::active_cell_iterator
                 cell = triangulation.begin_active();
         cell != triangulation.end(); ++cell) {
        SymmetricTensor<2, dim> accumulated_disl_stress;
        SymmetricTensor<2, dim> accumulated_image_stress;
        SymmetricTensor<2, dim> accumulated_full_stress;
        SymmetricTensor<2, dim> accumulated_strain;
        for (int q = 0; q < quadrature_formula.size(); ++q) {
            accumulated_disl_stress +=
                    reinterpret_cast<UserData<dim> *>
                    (cell->user_pointer())[q].dislocation_stress;
            accumulated_image_stress +=
                    reinterpret_cast<UserData<dim> *>
                    (cell->user_pointer())[q].image_stress;
            accumulated_full_stress +=
                    reinterpret_cast<UserData<dim> *>
                    (cell->user_pointer())[q].full_stress;
            accumulated_strain +=
                    reinterpret_cast<UserData<dim> *>
                    (cell->user_pointer())[q].strain;
        }
        image_stress11(cell->active_cell_index()) =
                (accumulated_image_stress / quadrature_formula.size())[0][0];
        image_stress12(cell->active_cell_index()) =
                (accumulated_image_stress / quadrature_formula.size())[0][1];
        image_stress22(cell->active_cell_index()) =
                (accumulated_image_stress / quadrature_formula.size())[1][1];
        disl_stress11(cell->active_cell_index()) =
                (accumulated_disl_stress / quadrature_formula.size())[0][0];
        disl_stress12(cell->active_cell_index()) =
                (accumulated_disl_stress / quadrature_formula.size())[0][1];
        disl_stress22(cell->active_cell_index()) =
                (accumulated_disl_stress / quadrature_formula.size())[1][1];
        full_stress11(cell->active_cell_index()) =
                (accumulated_full_stress / quadrature_formula.size())[0][0];
        full_stress12(cell->active_cell_index()) =
                (accumulated_full_stress / quadrature_formula.size())[0][1];
        full_stress22(cell->active_cell_index()) =
                (accumulated_full_stress / quadrature_formula.size())[1][1];
        strain11(cell->active_cell_index()) =
                (accumulated_strain / quadrature_formula.size())[0][0];
        strain12(cell->active_cell_index()) =
                (accumulated_strain / quadrature_formula.size())[0][1];
        strain22(cell->active_cell_index()) =
                (accumulated_strain / quadrature_formula.size())[1][1];


    }
    data_out.add_data_vector (image_stress11, "image_sigma_11");
    data_out.add_data_vector (image_stress12, "image_sigma_12");
    data_out.add_data_vector (image_stress22, "image_sigma_22");
    data_out.add_data_vector (disl_stress11, "disl_sigma_11");
    data_out.add_data_vector (disl_stress12, "disl_sigma_12");
    data_out.add_data_vector (disl_stress22, "disl_sigma_22");
    data_out.add_data_vector (full_stress11, "full_sigma_11");
    data_out.add_data_vector (full_stress12, "full_sigma_12");
    data_out.add_data_vector (full_stress22, "full_sigma_22");
    data_out.add_data_vector (strain11, "strain_11");
    data_out.add_data_vector (strain12, "strain_12");
    data_out.add_data_vector (strain22, "strain_22");

    data_out.build_patches ();
    data_out.write_vtk (output);
}

template <int dim>
void ElasticProblem<dim>::SetupQuadratureUserData() {
    unsigned int n_cells = triangulation.n_active_cells();
    triangulation.clear_user_data();
    quadrature_user_data.resize(n_cells * quadrature_formula.size());

    unsigned int index = 0;
    for (typename Triangulation<dim>::active_cell_iterator
                 cell = triangulation.begin_active();
         cell != triangulation.end(); ++cell)
    {
        cell->set_user_pointer(&quadrature_user_data[index]);
        index += quadrature_formula.size();
    }
    Assert(index == quadrature_user_data.size(), ExcInternalError());
}


template <int dim>
void ElasticProblem<dim>::UpdateUserData() {
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_quadrature_points | update_values | update_gradients);
    std::vector< std::vector<Tensor<1, dim> > >
            image_displacement_grads(quadrature_formula.size(),
                                     std::vector<Tensor<1, dim> >(dim)),
            overall_displacement_grads(quadrature_formula.size(),
                                       std::vector<Tensor<1, dim> >(dim));


    for(typename DoFHandler<dim>::active_cell_iterator
                cell = dof_handler.begin_active();
        cell != dof_handler.end(); ++cell) {

        UserData<dim>* local_user_data =
                reinterpret_cast<UserData<dim>* >(cell->user_pointer());
        Assert(local_user_data >= &quadrature_user_data.front(),
               ExcInternalError());
        Assert(local_user_data < &quadrature_user_data.back(),
               ExcInternalError());
        fe_values.reinit(cell);
        fe_values.get_function_gradients(image_solution,
                                         image_displacement_grads);
        fe_values.get_function_gradients(overall_solution,
                                         overall_displacement_grads);

        for (int q = 0; q < quadrature_formula.size(); ++q) {
            // Stress due to dislocations
            std::vector<SymmetricTensor<2,dim> > stress_from_dislocations(number_of_slip_systems);
            for (int k = 0; k < number_of_slip_systems; ++k) {
                for (size_t i = 0; i < dislocations[k].size(); ++i) {
                    stress_from_dislocations[k] += dislocations[k][i].getStress
                            (fe_values.quadrature_point(q), elastic);
                }
            }

            for (int k = 0; k < number_of_slip_systems; ++k) {
                if (abs(slip_system_angles[k] - 0.0) < ERROR) {
                    local_user_data[q].dislocation_stress += stress_from_dislocations[k];
                }
                else {
                    local_user_data[q].dislocation_stress +=
                             symmetrize(rotation_matrices[k]*
                             static_cast<Tensor<2, dim> >(stress_from_dislocations[k])*
                             transpose(rotation_matrices[k]));
                }
            }

            // Image stress
            local_user_data[q].image_stress =
                    stress_strain_tensor * get_strain(image_displacement_grads[q]);

            // Full stress
            local_user_data[q].full_stress = local_user_data[q].image_stress +
                                             local_user_data[q].dislocation_stress;

            //overall strain
            local_user_data[q].strain =
                    get_strain(overall_displacement_grads[q]);
        }
    }

}

template <int dim>
void ElasticProblem<dim>::ComputeBoundaryForceSolution() {
    FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                      update_values   | update_normal_vectors |
                                      update_quadrature_points | update_JxW_values);
    const unsigned int n_face_q_points = face_quadrature_formula.size();
    // loop over all cells:
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
            endc = dof_handler.end();
    for (; cell!=endc; ++cell) {
        for (unsigned int face_number = 0;
             face_number < GeometryInfo<dim>::faces_per_cell; ++face_number) {
            if (cell->face(face_number)->at_boundary() &&
                cell->face(face_number)->boundary_id() == 0) {
                UserData<dim>* local_user_data =
                        reinterpret_cast<UserData<dim>* >(cell->user_pointer());
                fe_face_values.reinit(cell, face_number);
                // quadrature
                for(unsigned int q = 0; q < n_face_q_points; ++q) {
                    Tensor<1, dim> force = local_user_data[q].full_stress*
                                       fe_face_values.normal_vector(q);
                    force_at_boundary_solution.push_back(
                            std::make_pair(fe_face_values.quadrature_point(q), force));
                }
            }
        }
    }
}

template <int dim>
void ElasticProblem<dim>::Run()
{
    using namespace std;
    CreateGrid(1e-5, 6);
    Tensor<1, dim> force; Tensor<1, dim> u;
    force[0] = 0.0; force[1] = 0.0;
    u[0] = 0.0; u[1] = 0.0;
    SetupBoundaryConditions(force, u);
    const vector<double> slip_angles = {0.0};
    SetupSlipSystems(slip_angles);
    double dist = 2000*burgers;
    /*vector<pair<Point<dim>, Sign> > points = {{Point<dim>(0, -2*dist), POSITIVE},
                                         {Point<dim>(0, -dist), POSITIVE},
                                         {Point<dim>(0, 0), POSITIVE},
                                         {Point<dim>(0, dist), POSITIVE},
                                         {Point<dim>(0, 2*dist), POSITIVE}};
                                         */
    vector<Point<dim> > points = {Point<dim>(0, 5.0e-6), Point<dim>(0, -5.0e-6)};
    vector<Sign> signs = {POSITIVE, NEGATIVE};
    vector<double> angles = {0.0, 0.0};

    CreateDislocations(points, signs, angles);

    for (int k = 0; k < dislocations.size(); ++k) {
            std::cout << " Dislocation number on " << k << " slip plane is " <<
                                     slip_system_sizes[k] << "\n";
    }

    std::cout << "   Number of active cells:       "
              << triangulation.n_active_cells()
              << std::endl;

    SetupSystem();
    SetupElasticProperties (/*lambda*/ 60*1e9, /*mu*/ 27*1e9);
    SetupQuadratureUserData();

    std::cout << "   Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl;

    AssembleSystemTensor();
    Solve();
    AddDislocationComponentToResult();
    UpdateUserData();
    //ComputeBoundaryForceSolution();

    OutputResults(0);
}




#endif //ELASTIC_PROBLEM_H

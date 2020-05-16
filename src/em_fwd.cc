#include "em_utils.h"
#include "em_ctx.h"
#include "em_io.h"
#include "em_topo.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/timer.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <fstream>
#include <iostream>
#include <set>
#include <string>

template <typename CellType>
size_t get_rho_index(EMContext *ctx, const CellType &cell) {
  (void)ctx;
  return cell->user_index();
}

template <typename CellType>
double get_cell_sigma(EMContext *ctx, const CellType &cell) {
  return 1.0 / ctx->rho[get_rho_index(ctx, cell)];
}

void setup_system(EMContext *ctx, int cycle) {
  if (cycle == 0) {
    ctx->dh->clear();
    ctx->dh->initialize(*ctx->mesh, *ctx->fe);
  } else {
    ctx->dh->distribute_dofs(*ctx->fe);
  }

  ctx->ac->clear();
  dealii::DoFTools::make_hanging_node_constraints(*ctx->dh, *ctx->ac);
  ctx->ac->close();

  std::cout << "dof_handler n_dofs:" << ctx->dh->n_dofs()<<std::endl;
  std::cout << "           n_cells:" << ctx->mesh->n_active_cells()<<std::endl;
  dealii::DynamicSparsityPattern dsp(ctx->dh->n_dofs(), ctx->dh->n_dofs());
  dealii::DoFTools::make_sparsity_pattern(*ctx->dh, dsp, *ctx->ac, false);
  ctx->sp->copy_from(dsp);

  ctx->K->reinit(*ctx->sp);
  ctx->u->reinit(ctx->dh->n_dofs());
  ctx->s->reinit(ctx->dh->n_dofs());

  ctx->error->reinit(ctx->mesh->n_active_cells());
  ctx->smoothness->reinit(ctx->mesh->n_active_cells());

  dealii::deallog << "Cycle: " << cycle << std::endl;
  dealii::deallog << "DoFs: " << ctx->dh->n_dofs() << std::endl;
}

void assemble_system(EMContext *ctx, int tidx) {
  double foo, sigma;
  Point ps = ctx->tx[tidx];
  dealii::Tensor<1, 3> n, r;
  size_t i, j, f, q_point, dofs_per_cell;

  dealii::Vector<double> cell_rhs;
  dealii::FullMatrix<double> cell_matrix;
  DoFHandler::active_cell_iterator cell, tx_cell;
  std::vector<dealii::types::global_dof_index> local_dof_indices;
  std::pair<Triangulation::active_cell_iterator, Point> cell_point;

  dealii::hp::FEValues<3> hp_fe_values(*ctx->fe, *ctx->quadratures,
                                         dealii::update_values | dealii::update_gradients |
                                         dealii::update_quadrature_points | dealii::update_JxW_values);
  dealii::hp::FEFaceValues<3> hp_fe_face_values(*ctx->fe, *ctx->face_quadratures,
                                                  dealii::update_values | dealii::update_quadrature_points |
                                                  dealii::update_normal_vectors | dealii::update_JxW_values);

  cell_point = ctx->tria_finder->find_active_cell_around_point(ps);
  tx_cell = DoFHandler::active_cell_iterator(*(cell_point.first), ctx->dh.get());

  for (cell = ctx->dh->begin_active(); cell != ctx->dh->end(); ++cell) {
    dofs_per_cell = cell->get_fe().dofs_per_cell;

    cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
    cell_matrix = 0;

    hp_fe_values.reinit(cell);

    const dealii::FEValues<3> &fe_values = hp_fe_values.get_present_fe_values();

    sigma = get_cell_sigma(ctx, cell);

    for (q_point = 0; q_point < fe_values.n_quadrature_points; ++q_point) {
      for (i = 0; i < dofs_per_cell; ++i) {
        for (j = 0; j < dofs_per_cell; ++j) {
          cell_matrix(i, j) += (sigma * fe_values.shape_grad(i, q_point) * fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point));
        }
      }
    }

    for (f = 0; f < dealii::GeometryInfo<3>::faces_per_cell; ++f) {
      auto face = cell->face(f);
      if (face->at_boundary()) {
        hp_fe_face_values.reinit(cell, f);
        const dealii::FEFaceValues<3> &fe_face_values = hp_fe_face_values.get_present_fe_values();

        if (fe_face_values.get_normal_vectors()[0][2] < 0) {
          continue;
        }

        r = face->center() - ps;
        for (q_point = 0; q_point < fe_face_values.n_quadrature_points; ++q_point) {
          n = fe_face_values.normal_vector(q_point);
          foo = dealii::scalar_product(r, n) / (r.norm() * r.norm() * n.norm()) * sigma;
          for (i = 0; i < dofs_per_cell; ++i) {
            for (j = 0; j < dofs_per_cell; ++j) {
              cell_matrix(i, j) += (foo * fe_face_values.shape_value(i, q_point) * fe_face_values.shape_value(j, q_point) * fe_face_values.JxW(q_point));
            }
          }
        }
      }
    }

    local_dof_indices.resize(dofs_per_cell);
    cell->get_dof_indices(local_dof_indices);

    ctx->ac->distribute_local_to_global(cell_matrix, local_dof_indices, *ctx->K);

    if (cell == tx_cell) {
      const dealii::Quadrature<3> quadrature(cell_point.second);
      dealii::FEValues<3> tx_fe_values(tx_cell->get_fe(), quadrature, dealii::update_values);

      tx_fe_values.reinit(tx_cell);

      cell_rhs.reinit(dofs_per_cell);
      cell_rhs = 0.0;

      for (i = 0; i < dofs_per_cell; ++i) {
        cell_rhs[i] = tx_fe_values.shape_value(i, 0);
      }
      ctx->ac->distribute_local_to_global(cell_rhs, local_dof_indices, *ctx->s);
    }
  }
  ctx->K->compress(dealii::VectorOperation::add);
  ctx->s->compress(dealii::VectorOperation::add);
}

void solve_system(EMContext *ctx) {
#if 1
  dealii::SolverControl solver_control(ctx->s->size(), 1e-8 * ctx->s->l2_norm(), true);
  dealii::SolverCG<> cg(solver_control);

  dealii::PreconditionSSOR<> preconditioner;
  preconditioner.initialize(*ctx->K, 1.2);

  cg.solve(*ctx->K, *ctx->u, *ctx->s, preconditioner);

  ctx->ac->distribute(*ctx->u);
#else
  dealii::SparseDirectUMFPACK A_direct;

  A_direct.initialize(*ctx->K);
  A_direct.vmult(*ctx->u, *ctx->s);

  ctx->ac->distribute(*ctx->u);
#endif
}

void estimate_error_and_smoothness(EMContext *ctx) {

  {  
    dealii::Vector<double>  u_sig(ctx->dh->n_dofs());
    u_sig = 0;
    unsigned int dofs_per_cell;
    double self_sig;
    DoFHandler::active_cell_iterator cell = ctx->dh->begin_active(), endc = ctx->dh->end();
    for (size_t index = 0; cell != endc; ++cell, ++index){
        dofs_per_cell = cell->get_fe().dofs_per_cell;
        std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);
	self_sig = get_cell_sigma(ctx,cell);
        cell->get_dof_indices(local_dof_indices);
	if (self_sig == 0.01){
	   for (unsigned int i = 0; i < dofs_per_cell; i++){
	       if (u_sig(local_dof_indices[i])==0){
	          u_sig(local_dof_indices[i]) = (*ctx->u)(local_dof_indices[i]) * self_sig; 
	       }
	   } 
	}
    }
    //DoFHandler::active_cell_iterator cell = ctx->dh->begin_active(), endc = ctx->dh->end();
    for (size_t index = 0; cell != endc; ++cell, ++index){
        dofs_per_cell = cell->get_fe().dofs_per_cell;
        std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);
	self_sig = get_cell_sigma(ctx,cell);
        cell->get_dof_indices(local_dof_indices);
	if (self_sig == 0.1){
	   for (unsigned int i = 0; i < dofs_per_cell; i++){
	       if (u_sig(local_dof_indices[i])==0){
	          u_sig(local_dof_indices[i]) = (*ctx->u)(local_dof_indices[i]) * self_sig; 
	       }
	   } 
	}
    }

        
    
   
 
   
  dealii::KellyErrorEstimator<3>::estimate(*ctx->dh, *ctx->face_quadratures,
      std::map<dealii::types::boundary_id, const dealii::Function<3, double> *>(),
      u_sig, *ctx->error);
  float max_error;
  max_error = (*ctx->error).linfty_norm(); 
  (*ctx->error) /= max_error; 
  }
  
/*
  dealii::KellyErrorEstimator<3>::estimate(*ctx->dh, *ctx->face_quadratures,
      std::map<dealii::types::boundary_id, const dealii::Function<3, double> *>(),
      *ctx->u, *ctx->error);
*/
  const size_t N = ctx->max_order;

  std::vector<dealii::Tensor<1, 3> > k_vectors;
  std::vector<size_t> k_vectors_magnitude;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      for (size_t k = 0; k < N; ++k) {
        if (!((i == 0) && (j == 0) && (k == 0)) && (i * i + j * j + k * k < N * N)) {
          k_vectors.push_back(dealii::Point<3>(dealii::numbers::PI * i, dealii::numbers::PI * j, dealii::numbers::PI * k));
          k_vectors_magnitude.push_back(i * i + j * j + k * k);
        }
      }
    }
  }

  const size_t n_fourier_modes = k_vectors.size();
  std::vector<double> ln_k(n_fourier_modes);
  for (size_t i = 0; i < n_fourier_modes; ++i) {
    ln_k[i] = std::log(k_vectors[i].norm());
  }

  std::vector<dealii::Table<2, std::complex<double> > > fourier_transform_matrices(ctx->fe->size());

  dealii::QGauss<1> base_quadrature(2);
  dealii::QIterated<3> quadrature(base_quadrature, N);

  for (size_t fe = 0; fe < ctx->fe->size(); ++fe) {
    fourier_transform_matrices[fe].reinit(n_fourier_modes, (*ctx->fe)[fe].dofs_per_cell);

    for (size_t k = 0; k < n_fourier_modes; ++k)
      for (size_t j = 0; j < (*ctx->fe)[fe].dofs_per_cell; ++j) {
        std::complex<double> sum = 0;
        for (size_t q = 0; q < quadrature.size(); ++q) {
          const dealii::Point<3> x_q = quadrature.point(q);
          sum += std::exp(std::complex<double>(0, 1) * (k_vectors[k] * x_q)) *
                 (*ctx->fe)[fe].shape_value(j, x_q) * quadrature.weight(q);
        }
        fourier_transform_matrices[fe](k, j) = sum / std::pow(2 * dealii::numbers::PI, 1.0 * 3 / 2);
      }
  }

  std::vector<std::complex<double> > fourier_coefficients(n_fourier_modes);
  dealii::Vector<double> local_dof_values;

  // Then here is the loop:
  DoFHandler::active_cell_iterator cell = ctx->dh->begin_active(), endc = ctx->dh->end();
  for (size_t index = 0; cell != endc; ++cell, ++index) {
    local_dof_values.reinit(cell->get_fe().dofs_per_cell);
    cell->get_dof_values(*ctx->u, local_dof_values);

    for (size_t f = 0; f < n_fourier_modes; ++f) {
      fourier_coefficients[f] = 0;

      for (size_t i = 0; i < cell->get_fe().dofs_per_cell; ++i)
        fourier_coefficients[f] += fourier_transform_matrices[cell->active_fe_index()](f, i) * local_dof_values(i);
    }

    std::map<size_t, double> k_to_max_U_map;
    for (size_t f = 0; f < n_fourier_modes; ++f) {
      if ((k_to_max_U_map.find(k_vectors_magnitude[f]) == k_to_max_U_map.end()) || (k_to_max_U_map[k_vectors_magnitude[f]] < std::abs(fourier_coefficients[f]))) {
        k_to_max_U_map[k_vectors_magnitude[f]] = std::abs(fourier_coefficients[f]);
      }
    }
    double sum_1 = 0, sum_ln_k = 0, sum_ln_k_square = 0, sum_ln_U = 0, sum_ln_U_ln_k = 0;
    for (size_t f = 0; f < n_fourier_modes; ++f)
      if (k_to_max_U_map[k_vectors_magnitude[f]] == std::abs(fourier_coefficients[f])) {
        sum_1 += 1;
        sum_ln_k += ln_k[f];
        sum_ln_k_square += ln_k[f] * ln_k[f];
        sum_ln_U += std::log(std::abs(fourier_coefficients[f]));
        sum_ln_U_ln_k += std::log(std::abs(fourier_coefficients[f])) * ln_k[f];
      }

    const double mu = (1.0 / (sum_1 * sum_ln_k_square - sum_ln_k * sum_ln_k) * (sum_ln_k * sum_ln_U - sum_1 * sum_ln_U_ln_k));

    (*ctx->smoothness)(index) = mu - 1.0 * 3 / 2;
  }
}

void save_mesh(EMContext *ctx, int tidx, int cycle) {
  dealii::Vector<double> fe_degrees(ctx->mesh->n_active_cells()), rho(ctx->mesh->n_active_cells());

  DoFHandler::active_cell_iterator cell = ctx->dh->begin_active(), endc = ctx->dh->end();
  for (size_t index = 0; cell != endc; ++cell, ++index) {
    rho(index) = 1.0 / get_cell_sigma(ctx, cell);
    fe_degrees(index) = (*ctx->fe)[cell->active_fe_index()].degree;
  }

  dealii::DataOut<3, DoFHandler> data_out;

  data_out.attach_dof_handler(*ctx->dh);
  data_out.add_data_vector(*ctx->u, "u");
  data_out.add_data_vector(*ctx->error, "error");
  data_out.add_data_vector(*ctx->smoothness, "smoothness");
  data_out.add_data_vector(rho, "rho");
  data_out.add_data_vector(fe_degrees, "fe_degree");
  data_out.build_patches();

  std::ofstream of(string_format("%s-%02d-%02d.vtu", ctx->oprefix.c_str(), tidx, cycle));
  data_out.write_vtu(of);
}

void refine_receiving_area(EMContext *ctx, int tidx) {
  for (int i = 0; i < ctx->n_global_refinements; ++i) {
    ctx->mesh->set_all_refine_flags();
    ctx->mesh->execute_coarsening_and_refinement();
  }

  std::set<int> rx_indices;
  for (int i = 0; i < (int)ctx->otype.size(); ++i) {
    if (ctx->aidx[i] == tidx || ctx->bidx[i] == tidx) {
      if (ctx->midx[i] >= 0) {
        rx_indices.insert(ctx->midx[i]);
      }
      if (ctx->nidx[i] >= 0) {
        rx_indices.insert(ctx->nidx[i]);
      }
    }
  }

  std::vector<Point> pts(rx_indices.size() + 1);
  for (auto it = rx_indices.begin(); it != rx_indices.end(); ++it) {
    pts.push_back(ctx->rx[*it]);
  }
  pts.push_back(ctx->tx[tidx]);

  for (int r = 0; r < ctx->n_rx_cell_refinements; ++r) {
    ctx->tria_finder->reinit(*ctx->mesh);
    for (int i = 0; i < (int)pts.size(); ++i) {
      auto cell = ctx->tria_finder->find_active_cell_around_point(pts[i]).first;

      if (cell.state() != dealii::IteratorState::valid) {
        error(string_format("Point(%g, %g, %g) not found in triangulation.", pts[i][0], pts[i][1], pts[i][2]).c_str());
      }

      if (cell->diameter() > ctx->max_rx_cell_size) {
        cell->set_refine_flag();
      }
    }

    ctx->mesh->execute_coarsening_and_refinement();
  }

  for (auto cell = ctx->mesh->begin(0); cell != ctx->mesh->end(0); ++cell) {
    cell->recursively_set_user_index(cell->user_index());
  }
}

void refine_mesh(EMContext *ctx) {
  if(ctx->refine_fraction==0){
    DoFHandler::active_cell_iterator cell = ctx->dh->begin_active(), endc = ctx->dh->end();
    for (size_t index = 0; cell != endc; ++cell, ++index) {
        cell->set_active_fe_index(cell->active_fe_index() + 1);
    }
  }
  else{
  dealii::GridRefinement::refine_and_coarsen_fixed_number(*ctx->mesh, *ctx->error, ctx->refine_fraction, 0.0);

  float max_smoothness = *std::min_element(ctx->smoothness->begin(), ctx->smoothness->end()),
        min_smoothness = *std::max_element(ctx->smoothness->begin(), ctx->smoothness->end());
  {
    DoFHandler::active_cell_iterator cell = ctx->dh->begin_active(), endc = ctx->dh->end();
    for (size_t index = 0; cell != endc; ++cell, ++index) {
      if (cell->refine_flag_set()) {
        max_smoothness = std::max(max_smoothness, (*ctx->smoothness)(index));
        min_smoothness = std::min(min_smoothness, (*ctx->smoothness)(index));
      }
    }
  }

  const float threshold_smoothness =  7 * (max_smoothness + min_smoothness) / 11;
  std::cout << "max:" << max_smoothness << "min" << min_smoothness << std::endl; 
  //const float threshold_smoothness =  (max_smoothness + min_smoothness) / 2;
  std::cout << "thr:" << threshold_smoothness << std::endl;
  {
    int i = 0;
    DoFHandler::active_cell_iterator cell = ctx->dh->begin_active(), endc = ctx->dh->end();
    for (size_t index = 0; cell != endc; ++cell, ++index) {
      if (cell->refine_flag_set() && ((*ctx->smoothness)(index) > threshold_smoothness) && ((cell->active_fe_index() + 1) < ctx->fe->size())) {
        cell->clear_refine_flag();
        cell->set_active_fe_index(cell->active_fe_index() + 1);
	i += 1;
      }
    }

/*
    DoFHandler::active_cell_iterator cell = ctx->dh->begin_active(), endc = ctx->dh->end();
    for (size_t index = 0; cell != endc; ++cell, ++index) {
      double self_sig, left_sig, right_sig, front_sig, back_sig, down_sig, up_sig; 
      if (cell->refine_flag_set()){
	if(!(cell->at_boundary())){ 
	  self_sig = get_cell_sigma(ctx,cell);
	  left_sig = get_cell_sigma(ctx,cell->neighbor(0));
	  right_sig = get_cell_sigma(ctx,cell->neighbor(1));
	  front_sig = get_cell_sigma(ctx,cell->neighbor(2));
	  back_sig = get_cell_sigma(ctx,cell->neighbor(3));
	  down_sig = get_cell_sigma(ctx,cell->neighbor(4));
	  up_sig = get_cell_sigma(ctx,cell->neighbor(5));
	  if((self_sig==left_sig) &&(self_sig==right_sig)&&(self_sig == front_sig)&&(self_sig==back_sig)&&(self_sig=down_sig)&&(self_sig==up_sig) && ((*ctx->smoothness)(index)>
			 threshold_smoothness) && ((cell->active_fe_index()+1)<ctx->fe->size())){
	     cell->clear_refine_flag();
	     cell->set_active_fe_index(cell->active_fe_index()+1);
	     i += 1;
	 
	  }
	}
      
       else{ 
	 if(((*ctx->smoothness)(index) > threshold_smoothness) && ((cell->active_fe_index()+1)<ctx->fe->size())){
            cell->clear_refine_flag(); 
	    cell->set_active_fe_index(cell->active_fe_index()+1);
         }
       }
     }
    }
 */
    std::cout << "i======" << i << std::endl;
  }

  ctx->mesh->execute_coarsening_and_refinement();

  for (auto cell = ctx->mesh->begin(0); cell != ctx->mesh->end(0); ++cell) {
    cell->recursively_set_user_index(cell->user_index());
  }
  }
}

void interpolate_value(EMContext *ctx, const Point &p, double &u) {
  auto cell_point = ctx->tria_finder->find_active_cell_around_point(p);
  DoFHandler::active_cell_iterator rx_cell = DoFHandler::active_cell_iterator(*(cell_point.first), ctx->dh.get());

  const dealii::Quadrature<3> quadrature(cell_point.second);
  dealii::FEValues<3> rx_fe_values(rx_cell->get_fe(), quadrature, dealii::update_values);

  int dofs_per_cell = rx_cell->get_fe().dofs_per_cell;

  std::vector<dealii::types::global_dof_index> dof_indices(dofs_per_cell);
  rx_cell->get_dof_indices(dof_indices);

  rx_fe_values.reinit(rx_cell);

  u = 0.0;
  for (int i = 0; i < dofs_per_cell; ++i) {
    u += rx_fe_values.shape_value(i, 0) * (*ctx->u)[dof_indices[i]];
  }
}

void calculate_response(EMContext *ctx, int tidx) {
	   std::cout << "calculate_response start !!!" << std::endl;
  double m, n, k;

  for (int i = 0; i < (int)ctx->otype.size(); ++i) {
    if (ctx->otype[i] == 2 && ctx->aidx[i] == tidx) {
      interpolate_value(ctx, ctx->rx[ctx->midx[i]], m);
      ctx->rsp_rho[i] = 2.0 * dealii::numbers::PI * ctx->tx[ctx->aidx[i]].distance(ctx->rx[ctx->midx[i]]) * m;
    } else if (ctx->otype[i] == 4 && (ctx->aidx[i] == tidx || ctx->bidx[i] == tidx)) {
      k = ((ctx->tx[ctx->aidx[i]].distance(ctx->rx[ctx->midx[i]]) * ctx->tx[ctx->aidx[i]].distance(ctx->rx[ctx->nidx[i]]) *
            ctx->tx[ctx->bidx[i]].distance(ctx->rx[ctx->midx[i]]) * ctx->tx[ctx->bidx[i]].distance(ctx->rx[ctx->nidx[i]])) /
           (ctx->tx[ctx->aidx[i]].distance(ctx->rx[ctx->midx[i]]) * ctx->tx[ctx->aidx[i]].distance(ctx->rx[ctx->nidx[i]]) +
            ctx->tx[ctx->bidx[i]].distance(ctx->rx[ctx->midx[i]]) * ctx->tx[ctx->bidx[i]].distance(ctx->rx[ctx->nidx[i]]))) *
          (2.0 * dealii::numbers::PI / ctx->rx[ctx->midx[i]].distance(ctx->rx[ctx->nidx[i]]));

      if (ctx->bidx[i] == tidx) {
        k *= -1.0;
      }
      interpolate_value(ctx, ctx->rx[ctx->midx[i]], m);
      interpolate_value(ctx, ctx->rx[ctx->nidx[i]], n);

      ctx->rsp_rho[i] += k * (m - n);
    }
  }
  std::cout << "calculate rsp end!!" << std::endl;
}

void em_forward(EMContext *ctx) {
  FILE *fp; 
  //const char *f = string_format("%s-mesh.txt",ctx->oprefix.c_str()).c_str(); 
  fp = fopen(string_format("%s-mesh.txt",ctx->oprefix.c_str()).c_str(),"w");
  int tidx, cycle;

  for (tidx = 0; tidx < (int)ctx->tx.size(); ++tidx) {
    dealii::Timer timer;
    ctx->mesh->clear();
    ctx->mesh->copy_triangulation(*ctx->coarse_mesh);
    refine_receiving_area(ctx, tidx);

    for (cycle = 0; cycle < ctx->max_adaptive_refinements; ++cycle) {
      ctx->tria_finder->reinit(*ctx->mesh);
      setup_system(ctx, cycle);

      fprintf(fp,"%d  %d\n",ctx->dh->n_dofs(),ctx->mesh->n_active_cells());
      assemble_system(ctx, tidx);
      solve_system(ctx);

      calculate_response(ctx, tidx);
      std::cout << "time=" << timer.cpu_time() << std::endl;
      fprintf(fp,"%f\n",timer.cpu_time());

      save_rsp(ctx,(string_format("%s-%02d-%d.rsp",ctx->oprefix.c_str(),tidx,cycle).c_str()));

      estimate_error_and_smoothness(ctx);
      save_mesh(ctx, tidx, cycle);
      if (cycle == (ctx->max_adaptive_refinements - 1)) {
        break;
      }
      refine_mesh(ctx);

    }
    //calculate_response(ctx,tidx);
   // fclose(fp);

  }
  //save_rsp(ctx,(ctx->oprefix+".rsp").c_str());

}

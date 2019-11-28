#ifndef _EM_CTX_H_
#define _EM_CTX_H_ 1

#include "em_defs.h"
#include "em_mesh.h"


#include <deal.II/base/parameter_handler.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <memory>
#include <string>
#include <vector>
#include <fstream>

class TopographyFunction;

struct EMContext {
  std::vector<Point> tx, rx;
  std::vector<int> otype, aidx, bidx, midx, nidx;
  std::vector<double> rsp_rho, rsp_eta;

  dealii::Vector<double> rho, eta;
  dealii::Vector<double> rho_lb, rho_ub;
  std::shared_ptr<TriaFinder> tria_finder;
  std::shared_ptr<Triangulation> coarse_mesh, mesh;

  std::shared_ptr<dealii::hp::FECollection<3>> fe;
  std::shared_ptr<dealii::hp::QCollection<3>> quadratures;
  std::shared_ptr<dealii::hp::QCollection<2>> face_quadratures;
  std::shared_ptr<DoFHandler> dh;
  dealii::Vector<double> rho_idx;

  int init_order, max_order, mapping_order;
  std::shared_ptr<AffineConstraints> ac;
  std::shared_ptr<dealii::SparsityPattern> sp;

  std::shared_ptr<dealii::SparseMatrix<double>> K;
  std::shared_ptr<dealii::Vector<double>> u, s, dual_u;
  std::shared_ptr<dealii::Vector<float>> error, smoothness;

  std::shared_ptr<dealii::Mapping<3>> mapping;

  std::shared_ptr<TopographyFunction> tf;

  std::ofstream log_of;
  dealii::ParameterHandler prm;
  std::string iprefix, oprefix;
  double refine_fraction, max_tx_cell_size, max_rx_cell_size, min_cell_size, topo_tol;
  int max_adaptive_refinements, max_dofs, n_global_refinements, n_tx_cell_refinements,
      n_rx_cell_refinements, topo_type, refine_strategy, K_max_it, max_topo_refinements, mesh_format;
};

void declare_parameter(EMContext *);
void read_parameters(EMContext *, const std::string &);

void create_context(EMContext *);
void destroy_context(EMContext *);

#endif

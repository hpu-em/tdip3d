#include <deal.II/base/logstream.h>
#include <deal.II/fe/mapping_q_generic.h>

#include "em_ctx.h"
#include "em_utils.h"

void declare_parameter(EMContext *ctx) {
  ctx->prm.declare_entry("iprefix", "", dealii::Patterns::FileName(), "Prefix of the input files");
  ctx->prm.declare_entry("oprefix", "",
                         dealii::Patterns::FileName(dealii::Patterns::FileName::output),
                         "Prefix of the output files");
  ctx->prm.declare_entry("init_order","1",dealii::Patterns::Integer(1,5),"Initial order of Finite Element");
  ctx->prm.declare_entry("max_order", "5", dealii::Patterns::Integer(1, 5), "Maximum order of Finite Element");
  ctx->prm.declare_entry("max_adaptive_refinements", "1", dealii::Patterns::Integer(0, 20), "Maximum adaptive refinements");
  ctx->prm.declare_entry("refine_fraction", "0.3", dealii::Patterns::Double(0.0, 1.0), "Refine fraction");
  ctx->prm.declare_entry("n_global_refinements", "0", dealii::Patterns::Integer(0, 5), "Maximum global refinements");
  ctx->prm.declare_entry("n_rx_cell_refinements", "0", dealii::Patterns::Integer(0, 10), "Number of refinements at receiver point");
  ctx->prm.declare_entry("max_rx_cell_size", "-1.0", dealii::Patterns::Double(-1.0, 100), "Maximum cell size at receiver point");
}

void read_parameters(EMContext *ctx, const std::string &fn) {
  ctx->prm.parse_input(fn);
  ctx->iprefix = ctx->prm.get("iprefix");
  ctx->oprefix = ctx->prm.get("oprefix");
  ctx->init_order = ctx->prm.get_integer("init_order");
  ctx->max_order = ctx->prm.get_integer("max_order");
  ctx->max_adaptive_refinements = ctx->prm.get_integer("max_adaptive_refinements");
  ctx->refine_fraction = ctx->prm.get_double("refine_fraction");
  ctx->n_global_refinements = ctx->prm.get_integer("n_global_refinements");
  ctx->n_rx_cell_refinements = ctx->prm.get_integer("n_rx_cell_refinements");
  ctx->max_rx_cell_size = ctx->prm.get_double("max_rx_cell_size");
}

void create_context(EMContext *ctx) {
  int order;

  ctx->coarse_mesh.reset(new Triangulation);
  ctx->mesh.reset(new Triangulation);

  ctx->tria_finder.reset(new TriaFinder);

  ctx->mapping.reset(new dealii::MappingQGeneric<3>(1));

  ctx->ac.reset(new AffineConstraints);
  ctx->sp.reset(new dealii::SparsityPattern);
  ctx->K.reset(new dealii::SparseMatrix<double>);
  ctx->s.reset(new dealii::Vector<double>);
  ctx->u.reset(new dealii::Vector<double>);

  ctx->error.reset(new dealii::Vector<float>);
  ctx->smoothness.reset(new dealii::Vector<float>);

  ctx->fe.reset(new dealii::hp::FECollection<3>);
  ctx->quadratures.reset(new dealii::hp::QCollection<3>);
  ctx->face_quadratures.reset(new dealii::hp::QCollection<2>);
  ctx->dh.reset(new DoFHandler);

  for (order = ctx->init_order; order <= ctx->max_order; ++order) {
    ctx->fe->push_back(dealii::FE_Q<3>(order));
    ctx->quadratures->push_back(dealii::QGauss<3>(order + 1));
    ctx->face_quadratures->push_back(dealii::QGauss<2>(order + 1));
  }

  ctx->log_of.open((ctx->oprefix + ".log").c_str());
  dealii::deallog.attach(ctx->log_of, false);
  dealii::deallog.depth_file(2);
}

void destroy_context(EMContext *ctx) {
  ctx->K = NULL;
  ctx->s = NULL;
  ctx->u = NULL;

  ctx->ac = NULL;

  ctx->dh = NULL;
  ctx->fe = NULL;

  ctx->tria_finder = NULL;

  ctx->mesh = NULL;
  ctx->coarse_mesh = NULL;

  ctx->error = NULL;
  ctx->smoothness = NULL;

  ctx->mapping = NULL;
}

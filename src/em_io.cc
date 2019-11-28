#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_postprocessor.h>

#include <fstream>

#include "em_ctx.h"
#include "em_utils.h"

void read_mdl(EMContext *ctx, const char *fn) {
  std::string l;
  std::stringstream ss;
  std::ifstream ifs(fn);
  std::vector<Point> points;
  size_t n_points, n_cells, i, j;
  std::vector<dealii::CellData<3> > cells;
  Triangulation::active_cell_iterator cell;

  if (!ifs.good()) {
    error(string_format("Unable to open file %s.", fn).c_str());
  }

  while (std::getline(ifs, l)) {
    ss << parse_string(l);
  }

  ss >> n_points;
  points.resize(n_points);
  for (i = 0; i < n_points; ++i) {
    ss >> points[i][0] >> points[i][1] >> points[i][2];
  }

  ss >> n_cells;
  cells.resize(n_cells);
  for (i = 0; i < n_cells; ++i) {
    for (j = 0; j < 8; ++j) {
      ss >> cells[i].vertices[j];
      cells[i].manifold_id = i;
    }
  }

  ss >> n_cells;
  ctx->rho.reinit(n_cells);
  ctx->eta.reinit(n_cells);
  ctx->rho_lb.reinit(n_cells);
  ctx->rho_ub.reinit(n_cells);

  for (i = 0; i < n_cells; ++i) {
    ss >> ctx->rho[i] >> ctx->eta[i];
    ss >> ctx->rho_lb[i] >> ctx->rho_ub[i];
  }

  ctx->coarse_mesh->clear();
  ctx->coarse_mesh->create_triangulation(points, cells, dealii::SubCellData());

  for (cell = ctx->coarse_mesh->begin_active(); cell != ctx->coarse_mesh->end(); ++cell) {
    cell->set_user_index(cell->manifold_id());
    cell->set_manifold_id(dealii::numbers::flat_manifold_id);
  }
}

void read_emd(EMContext *ctx, const char *fn) {
  std::string l;
  std::stringstream ss;
  std::ifstream ifs(fn);
  size_t n_rxes, n_txes, n_obses, i;

  if (!ifs.good()) {
    error(string_format("Unable to open file %s.", fn).c_str());
  }

  while (std::getline(ifs, l)) {
    ss << parse_string(l);
  }

  ss >> n_txes;
  ctx->tx.resize(n_txes);
  for (i = 0; i < n_txes; ++i) {
    ss >> ctx->tx[i][0] >> ctx->tx[i][1] >> ctx->tx[i][2];
  }

  ss >> n_rxes;
  ctx->rx.resize(n_rxes);
  for (i = 0; i < n_rxes; ++i) {
    ss >> ctx->rx[i][0] >> ctx->rx[i][1] >> ctx->rx[i][2];
  }

  ss >> n_obses;
  ctx->otype.resize(n_obses);
  ctx->aidx.resize(n_obses);
  ctx->bidx.resize(n_obses);
  ctx->midx.resize(n_obses);
  ctx->nidx.resize(n_obses);
  ctx->rsp_rho.resize(n_obses);
  ctx->rsp_eta.resize(n_obses);

  for (i = 0; i < n_obses; ++i) {
    ss >> ctx->otype[i] >> ctx->aidx[i] >> ctx->midx[i] >> ctx->nidx[i] >> ctx->bidx[i] >> ctx->rsp_rho[i] >> ctx->rsp_eta[i];
  }
}

void save_rsp(EMContext *ctx, const char *fn) {

  FILE *fp;
  int i, ntx, nrx, nobs;

  fp = fopen(fn, "w");

  ntx = ctx->tx.size();
  fprintf(fp, "# transmitters\n");
  fprintf(fp, "%d\n", ntx);
  for (i = 0; i < ntx; ++i) {
    fprintf(fp, "% .4E % .4E % .4E\n", ctx->tx[i][0], ctx->tx[i][1], ctx->tx[i][2]);
  }

  nrx = ctx->rx.size();
  fprintf(fp, "# recievers\n");
  fprintf(fp, "%d\n", nrx);
  for (i = 0; i < nrx; ++i) {
    fprintf(fp, "% .4E %.4E % .4E\n", ctx->rx[i][0], ctx->rx[i][1], ctx->rx[i][2]);
  }

  nobs = ctx->otype.size();
  fprintf(fp, "# observations\n");
  fprintf(fp, "%d\n", nobs);
  fprintf(fp, "#%6s %6s %5s %7s %7s %13s %13s\n", "otype", "aidx", "midx", "nidx", "bidx", "rho", "eta");
  for (i = 0; i < nobs; ++i) {
    fprintf(fp, "% 7d % 6d % 5d % 7d % 7d % 13.3f % 13.3f\n", ctx->otype[i], ctx->aidx[i], ctx->midx[i], ctx->nidx[i], ctx->bidx[i], ctx->rsp_rho[i], ctx->rsp_eta[i]);
  }

  fclose(fp);
}

#include <fstream>

#include <deal.II/grid/grid_tools.h>

#include "em_ctx.h"
#include "em_utils.h"
#include "em_topo.h"

void AnalyticalFunction::read(const char *fn) {
  std::string l;
  std::stringstream ss;
  std::ifstream ifs(fn);
  std::vector<std::string> expressions(1);

  while (std::getline(ifs, l)) {
    ss << parse_string(l);
  }
  ss >> horizon_ >> bottom_range_;

  expressions[0] = ss.str().substr(ss.tellg());

  fp.initialize("x,y", expressions, std::map<std::string, double>());
}

void InterpolationFunction::read(const char *fn) {
  std::string l;
  int nx, ny, i, j;
  std::stringstream ss;
  std::ifstream ifs(fn);
  dealii::Table<2, double> vals;
  std::vector<double> x_coords, y_coords;

  while (std::getline(ifs, l)) {
    ss << parse_string(l);
  }

  ss >> horizon_ >> bottom_range_;

  ss >> nx;
  x_coords.resize(nx);
  for (i = 0; i < nx; ++i) {
    ss >> x_coords[i];
  }

  ss >> ny;
  y_coords.resize(ny);
  for (i = 0; i < ny; ++i) {
    ss >> y_coords[i];
  }

  vals.reinit(dealii::TableIndices<2>(nx, ny));
  for (i = 0; i < nx; ++i) {
    for (j = 0; j < ny; ++j) {
      ss >> vals(dealii::TableIndices<2>(i, j));
    }
  }

  interp_.reset(new dealii::Functions::InterpolatedTensorProductGridData<2>({ x_coords, y_coords }, vals));
}

static Point transform(const std::function<double(double, double)> &offset, double horizon, double bottom_range, const Point &p) {
  Point q = p;
  double scale;

  if (p[2] <= (horizon + bottom_range)) {
    scale = ((horizon + bottom_range) - p[2]) / bottom_range;
    q[2] = p[2] + scale * offset(p[0], p[1]);
  }

  return q;
}

void adapt_topography(EMContext *ctx) {
  Point c;
  size_t cycle, f;
  double horizon, bottom_range;
  Triangulation::cell_iterator cell;
  std::function<double(double, double)> offset;
  Triangulation::active_cell_iterator active_cell;

  if (ctx->topo_type == Analytical) {
    ctx->tf.reset(new AnalyticalFunction(string_format("%s-topo.dat", ctx->iprefix.c_str()).c_str()));
  } else if (ctx->topo_type == Interp) {
    ctx->tf.reset(new InterpolationFunction(string_format("%s-topo.dat", ctx->iprefix.c_str()).c_str()));
  } else {
    ctx->tf.reset(new TopographyFunction);
  }

  horizon = ctx->tf->horizon();
  bottom_range = ctx->tf->bottom_range();

  offset = std::bind(&TopographyFunction::offset, ctx->tf.get(), std::placeholders::_1, std::placeholders::_2);

  for (active_cell = ctx->coarse_mesh->begin_active(); active_cell != ctx->coarse_mesh->end(); ++active_cell) {
    c = active_cell->center();
    if (c[2] > horizon && c[2] < (horizon + bottom_range)) {
      active_cell->set_all_manifold_ids(1);
    } else {
      active_cell->set_all_manifold_ids(0);
    }
  }
  ctx->coarse_mesh->set_manifold(0, dealii::FlatManifold<3>());
  ctx->coarse_mesh->set_manifold(1, TopographyManifold(horizon, bottom_range, offset));

  dealii::GridTools::transform(std::bind(transform, offset, horizon, bottom_range, std::placeholders::_1), *ctx->coarse_mesh);

  for (cycle = 0; cycle < (size_t)ctx->max_topo_refinements; ++cycle) {
    for (active_cell = ctx->coarse_mesh->begin_active(); active_cell != ctx->coarse_mesh->end(); ++active_cell) {
      for (f = 0; f < dealii::GeometryInfo<3>::faces_per_cell; ++f) {
        if (active_cell->manifold_id() == 2 && !active_cell->face(f)->at_boundary() && active_cell->neighbor(f)->manifold_id() == 1) {
          c = active_cell->face(f)->center();
          if (std::abs(c[2] - (horizon + offset(c[0], c[1]))) > ctx->topo_tol) {
            active_cell->set_refine_flag();
            break;
          }
        }
      }
    }
    ctx->coarse_mesh->execute_coarsening_and_refinement();
  }

  for (cell = ctx->coarse_mesh->begin(0); cell != ctx->coarse_mesh->end(); ++cell) {
    cell->recursively_set_user_index(cell->user_index());
  }
}

#include "em_mesh.h"

#include <deal.II/grid/grid_tools.h>

#include <limits>
#include <queue>

void TriaFinder::reinit(const Triangulation &t) {
  size_t v;
  Triangulation::active_cell_iterator cell;

  mesh = &t;

  v_index.reset(new KDTree(3, *this, 100));
  v_index->buildIndex();

  v_to_c.resize(mesh->n_vertices());

  for (cell = mesh->begin_active(); cell != mesh->end(); ++cell) {
    for (v = 0; v < dealii::GeometryInfo<3>::vertices_per_cell; v++) {
      v_to_c[cell->vertex_index(v)] = cell;
    }
  }
}

int TriaFinder::find_closest_vertex(const Point &p) const {
  size_t idx;
  double dist;

  v_index->knnSearch(&p[0], 1, &idx, &dist);

  return idx;
}

std::pair<Triangulation::active_cell_iterator, Point> TriaFinder::find_active_cell_around_point(const Point &p) const {
  bool inside_cell;
  std::vector<bool> touched_cells;
  Triangulation::active_cell_iterator cell;
  std::queue<Triangulation::active_cell_iterator> q;
  std::vector<Triangulation::active_cell_iterator> neighbors;
  std::vector<Triangulation::active_cell_iterator>::iterator it;
  std::pair<Triangulation::active_cell_iterator, Point> cell_point;

  cell_point = std::make_pair(Triangulation::active_cell_iterator(), Point());

  touched_cells.resize(mesh->n_active_cells());
  std::fill(touched_cells.begin(), touched_cells.end(), false);

  inside_cell = false;
  q.push(v_to_c[find_closest_vertex(p)]);

  while (!q.empty()) {
    cell = q.front();
    q.pop();

    if (touched_cells[cell->active_cell_index()]) {
      continue;
    }
    touched_cells[cell->active_cell_index()] = true;

    try {
      inside_cell = dealii::GeometryInfo<3>::is_inside_unit_cell(mapping->transform_real_to_unit_cell(cell, p), EPS);
    } catch (const dealii::Mapping<3>::ExcTransformationFailed &) {
      inside_cell = false;
    }

    if (inside_cell) {
      cell_point = std::make_pair(cell, dealii::GeometryInfo<3>::project_to_unit_cell(mapping->transform_real_to_unit_cell(cell, p)));
      break;
    }

    dealii::GridTools::get_active_neighbors<Triangulation>(cell, neighbors);
    for (it = neighbors.begin(); it != neighbors.end(); ++it) {
      if (!touched_cells[(*it)->active_cell_index()]) {
        q.push(*it);
      }
    }
  }

  return cell_point;
}

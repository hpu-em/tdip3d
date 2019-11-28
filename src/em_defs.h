#ifndef _EM_DEFS_H_
#define _EM_DEFS_H_ 1

#include <tuple>

#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

typedef dealii::Point<3> Point;

typedef dealii::hp::DoFHandler<3> DoFHandler;
typedef dealii::AffineConstraints<double> AffineConstraints;

typedef dealii::Triangulation<3> Triangulation;

const double PI = 3.14159265358979323846;
const double EPS = 1E-6;

enum TopographyType { None = 0, Analytical = 1, Interp = 2 };

enum RefineStrategy { FixedNumber = 0, FixedFraction = 1 };

#define EM_ERR_USER -100000

#endif

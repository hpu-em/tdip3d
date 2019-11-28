#ifndef _EM_TOPO_H_
#define _EM_TOPO_H_ 1

#include <deal.II/base/function_lib.h>
#include <deal.II/base/function_parser.h>

#include <deal.II/grid/manifold_lib.h>

struct EMContext;

class TopographyFunction {
public:
  TopographyFunction() {
    horizon_ = 0.0;
    bottom_range_ = -1.0;
  }

  double horizon() const { return horizon_; }
  double bottom_range() const { return bottom_range_; }

  virtual double offset(double, double) const { return 0; }
  virtual ~TopographyFunction() {}

protected:
  double horizon_, bottom_range_;
};

class InterpolationFunction : public TopographyFunction {
public:
  InterpolationFunction(const char *fn) { read(fn); }
  virtual double offset(double x, double y) const { return interp_->value(dealii::Point<2>(x, y)); }

private:
  void read(const char *);

private:
  std::shared_ptr<dealii::Functions::InterpolatedTensorProductGridData<2> > interp_;
};

class AnalyticalFunction : public TopographyFunction {
public:
  AnalyticalFunction(const char *fn) { read(fn); }
  virtual double offset(double x, double y) const { return fp.value(dealii::Point<2>(x, y)); }

private:
  void read(const char *);

private:
  dealii::FunctionParser<2> fp;
};

class TopographyManifold : public dealii::ChartManifold<3, 3> {
public:
  TopographyManifold(double h, double r, const std::function<double(double, double)> &f)
      : horizon(h), range(r), tf(f) {}

  virtual Point push_forward(const Point &p) const {
    Point q = p;
    double scale;

    scale = ((horizon + range) - p[2]) / range;
    q[2] = p[2] + scale * tf(p[0], p[1]);

    return q;
  }

  virtual Point pull_back(const Point &p) const {
    double t;
    Point q = p;

    t = tf(p[0], p[1]);
    q[2] = (range * p[2] - (horizon + range) * t) / (range - t);

    return q;
  }

  virtual std::unique_ptr<dealii::Manifold<3, 3> > clone() const {
    return std::unique_ptr<dealii::Manifold<3, 3> >(new TopographyManifold(horizon, range, tf));
  }

private:
  double horizon, range;
  std::function<double(double, double)> tf;
};

void adapt_topography(EMContext *);

#endif

#include "em_ctx.h"
#include "em_io.h"
#include "em_fwd.h"

#include <deal.II/base/logstream.h>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Parameter file must be given." << std::endl;
    return 1;
  }

  EMContext ctx;

  declare_parameter(&ctx);
  read_parameters(&ctx, argv[1]);

  create_context(&ctx);

  read_mdl(&ctx, (ctx.iprefix + ".mdl").c_str());
  read_emd(&ctx, (ctx.iprefix + ".emd").c_str());

  em_forward(&ctx);

  destroy_context(&ctx);

  return 0;
}

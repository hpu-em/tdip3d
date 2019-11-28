SET(tdip3d_SOURCES em_utils.cc em_ctx.cc em_mesh.cc em_io.cc em_topo.cc em_fwd.cc em_main.cc)

ADD_EXECUTABLE(tdip3d ${tdip3d_SOURCES})
DEAL_II_SETUP_TARGET(tdip3d)

#ifndef _EM_IO_H_
#define _EM_IO_H_

#include "em_defs.h"

struct EMContext;

void read_mdl(EMContext *, const char *);
void read_emd(EMContext *, const char *);

void save_rsp(EMContext *, const char *);

#endif

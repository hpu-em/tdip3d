#ifndef _EM_UTILS_H_
#define _EM_UTILS_H_ 1

#include <string>
#include <string.h>

std::string string_format(const char *, ...);
std::string parse_string(const std::string &);
std::string base_name(const std::string &);

void error(const std::string &);

#endif

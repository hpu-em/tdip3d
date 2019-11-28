CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(tdip3d CXX)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

FIND_PACKAGE(deal.II 9.0 QUIET HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR})
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a sufficiently recent version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

INCLUDE_DIRECTORIES(${DEAL_II_INCLUDE_DIRS})

ADD_SUBDIRECTORY(src)

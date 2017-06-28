##
#  CMake script for the step-18 tutorial program:
##

# Set the name of the project and target:
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

SET(TARGET "ddisl")
SET(CMAKE_CXX_STANDART 11)
# Declare all source files the target consists of. Here, this is only
# the one step-X.cc file, but as you expand your project you may wish
# to add other source files as well. If your project becomes much larger,
# you may want to either replace the following statement by something like
#  FILE(GLOB_RECURSE TARGET_SRC  "source/*.cc")
#  FILE(GLOB_RECURSE TARGET_INC  "include/*.h")
#  SET(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC})
# or switch altogether to the large project CMakeLists.txt file discussed
# in the "CMake in user projects" page accessible from the "User info"
# page of the documentation.


FILE(GLOB_RECURSE TARGET_INC  "src/*.hpp")
INCLUDE_DIRECTORIES(src)

ADD_EXECUTABLE(${TARGET} ${TARGET}.cc ${TARGET_INC})

FIND_PACKAGE(deal.II 8.4 REQUIRED HINTS
    ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../  $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF ()


add_subdirectory(lib/googletest)
include_directories(lib/googletest/googletest/include)
add_executable(Tests tests/tests.cpp)
target_link_libraries(Tests gtest gtest_main)

DEAL_II_SETUP_TARGET(Tests)
DEAL_II_INITIALIZE_CACHED_VARIABLES()
#PROJECT(${TARGET})
DEAL_II_SETUP_TARGET(${TARGET})
#DEAL_II_INVOKE_AUTOPILOT()


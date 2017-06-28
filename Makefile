# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/home/clion/bin/cmake/bin/cmake

# The command to remove a file.
RM = /home/home/clion/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/home/projects/dislocation

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/home/projects/dislocation

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target install/local
install/local: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing only the local directory..."
	/home/home/clion/bin/cmake/bin/cmake -DCMAKE_INSTALL_LOCAL_ONLY=1 -P cmake_install.cmake
.PHONY : install/local

# Special rule for the target install/local
install/local/fast: install/local

.PHONY : install/local/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/home/home/clion/bin/cmake/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/home/home/clion/bin/cmake/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# Special rule for the target install
install: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Install the project..."
	/home/home/clion/bin/cmake/bin/cmake -P cmake_install.cmake
.PHONY : install

# Special rule for the target install
install/fast: preinstall/fast
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Install the project..."
	/home/home/clion/bin/cmake/bin/cmake -P cmake_install.cmake
.PHONY : install/fast

# Special rule for the target list_install_components
list_install_components:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Available install components are: \"Unspecified\""
.PHONY : list_install_components

# Special rule for the target list_install_components
list_install_components/fast: list_install_components

.PHONY : list_install_components/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/home/projects/dislocation/CMakeFiles /home/home/projects/dislocation/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/home/projects/dislocation/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named Tests

# Build rule for target.
Tests: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 Tests
.PHONY : Tests

# fast build rule for target.
Tests/fast:
	$(MAKE) -f CMakeFiles/Tests.dir/build.make CMakeFiles/Tests.dir/build
.PHONY : Tests/fast

#=============================================================================
# Target rules for targets named ddisl

# Build rule for target.
ddisl: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 ddisl
.PHONY : ddisl

# fast build rule for target.
ddisl/fast:
	$(MAKE) -f CMakeFiles/ddisl.dir/build.make CMakeFiles/ddisl.dir/build
.PHONY : ddisl/fast

#=============================================================================
# Target rules for targets named gmock_main

# Build rule for target.
gmock_main: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gmock_main
.PHONY : gmock_main

# fast build rule for target.
gmock_main/fast:
	$(MAKE) -f lib/googletest/googlemock/CMakeFiles/gmock_main.dir/build.make lib/googletest/googlemock/CMakeFiles/gmock_main.dir/build
.PHONY : gmock_main/fast

#=============================================================================
# Target rules for targets named gmock

# Build rule for target.
gmock: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gmock
.PHONY : gmock

# fast build rule for target.
gmock/fast:
	$(MAKE) -f lib/googletest/googlemock/CMakeFiles/gmock.dir/build.make lib/googletest/googlemock/CMakeFiles/gmock.dir/build
.PHONY : gmock/fast

#=============================================================================
# Target rules for targets named gtest_main

# Build rule for target.
gtest_main: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gtest_main
.PHONY : gtest_main

# fast build rule for target.
gtest_main/fast:
	$(MAKE) -f lib/googletest/googlemock/gtest/CMakeFiles/gtest_main.dir/build.make lib/googletest/googlemock/gtest/CMakeFiles/gtest_main.dir/build
.PHONY : gtest_main/fast

#=============================================================================
# Target rules for targets named gtest

# Build rule for target.
gtest: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gtest
.PHONY : gtest

# fast build rule for target.
gtest/fast:
	$(MAKE) -f lib/googletest/googlemock/gtest/CMakeFiles/gtest.dir/build.make lib/googletest/googlemock/gtest/CMakeFiles/gtest.dir/build
.PHONY : gtest/fast

ddisl.o: ddisl.cc.o

.PHONY : ddisl.o

# target to build an object file
ddisl.cc.o:
	$(MAKE) -f CMakeFiles/ddisl.dir/build.make CMakeFiles/ddisl.dir/ddisl.cc.o
.PHONY : ddisl.cc.o

ddisl.i: ddisl.cc.i

.PHONY : ddisl.i

# target to preprocess a source file
ddisl.cc.i:
	$(MAKE) -f CMakeFiles/ddisl.dir/build.make CMakeFiles/ddisl.dir/ddisl.cc.i
.PHONY : ddisl.cc.i

ddisl.s: ddisl.cc.s

.PHONY : ddisl.s

# target to generate assembly for a file
ddisl.cc.s:
	$(MAKE) -f CMakeFiles/ddisl.dir/build.make CMakeFiles/ddisl.dir/ddisl.cc.s
.PHONY : ddisl.cc.s

tests/tests.o: tests/tests.cpp.o

.PHONY : tests/tests.o

# target to build an object file
tests/tests.cpp.o:
	$(MAKE) -f CMakeFiles/Tests.dir/build.make CMakeFiles/Tests.dir/tests/tests.cpp.o
.PHONY : tests/tests.cpp.o

tests/tests.i: tests/tests.cpp.i

.PHONY : tests/tests.i

# target to preprocess a source file
tests/tests.cpp.i:
	$(MAKE) -f CMakeFiles/Tests.dir/build.make CMakeFiles/Tests.dir/tests/tests.cpp.i
.PHONY : tests/tests.cpp.i

tests/tests.s: tests/tests.cpp.s

.PHONY : tests/tests.s

# target to generate assembly for a file
tests/tests.cpp.s:
	$(MAKE) -f CMakeFiles/Tests.dir/build.make CMakeFiles/Tests.dir/tests/tests.cpp.s
.PHONY : tests/tests.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... install/local"
	@echo "... rebuild_cache"
	@echo "... edit_cache"
	@echo "... install"
	@echo "... list_install_components"
	@echo "... Tests"
	@echo "... ddisl"
	@echo "... gmock_main"
	@echo "... gmock"
	@echo "... gtest_main"
	@echo "... gtest"
	@echo "... ddisl.o"
	@echo "... ddisl.i"
	@echo "... ddisl.s"
	@echo "... tests/tests.o"
	@echo "... tests/tests.i"
	@echo "... tests/tests.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system


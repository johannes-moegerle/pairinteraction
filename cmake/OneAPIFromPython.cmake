# SPDX-FileCopyrightText: 2026 PairInteraction Developers
# SPDX-License-Identifier: LGPL-3.0-or-later

include_guard(GLOBAL)

# oneapi_from_python(<name> <devel-package> <config-file> <runtime-package> <library-stem>)
#
# Locate an Intel oneAPI library that is obtained from Python packages. The Python interpreter is asked for the package
# config file <config-file> of the package <devel-package> and for the library whose stem contains <library-stem> in the
# package <runtime-package>. If this succeeds, <name>_ROOT and <name>_DIR are set and the config directory is appended
# to CMAKE_PREFIX_PATH. In any case, ONEAPI_PYTHON is set to the interpreter that was used (empty if none was found).
function(oneapi_from_python name devel_package config_file runtime_package library_stem)
  # Prefer the interpreter that has already been found by the top-level project so that all dependencies that are
  # obtained from Python packages are taken from the same Python environment
  if(Python_EXECUTABLE)
    set(ONEAPI_PYTHON "${Python_EXECUTABLE}")
  else()
    find_package(
      Python3
      COMPONENTS Interpreter
      QUIET)
    set(ONEAPI_PYTHON "${Python3_EXECUTABLE}")
  endif()

  set(ONEAPI_PYTHON
      "${ONEAPI_PYTHON}"
      PARENT_SCOPE)

  if(NOT ONEAPI_PYTHON)
    message(STATUS "Python interpreter not found; skip discovering ${name} through the '${devel_package}' package.")
    return()
  endif()

  execute_process(
    COMMAND
      ${ONEAPI_PYTHON} -c "import sys
from importlib.metadata import files, PackageNotFoundError
devel_package, config_file, runtime_package, library_stem = sys.argv[1:]
try:
    config_path = next(p for p in files(devel_package) if config_file in p.name).locate().resolve()
    library_path = next(p for p in files(runtime_package) if library_stem in p.stem).locate().resolve()
    print(library_path.parent.parent, config_path, sep='|')
except PackageNotFoundError:
    sys.exit(1)"
      "${devel_package}" "${config_file}" "${runtime_package}" "${library_stem}"
    RESULT_VARIABLE ONEAPI_RESULT
    OUTPUT_VARIABLE ONEAPI_PATHS
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT ONEAPI_RESULT EQUAL 0)
    message(STATUS "Failed to find the '${devel_package}' Python package using ${ONEAPI_PYTHON}.")
    return()
  endif()

  string(REPLACE "|" ";" ONEAPI_PATHS_LIST "${ONEAPI_PATHS}")
  list(GET ONEAPI_PATHS_LIST 0 root)
  list(GET ONEAPI_PATHS_LIST 1 config_path)
  # The interpreter prints native paths, on Windows they contain backslashes that CMake cannot handle reliably
  cmake_path(SET root NORMALIZE "${root}")
  cmake_path(SET config_path NORMALIZE "${config_path}")
  cmake_path(GET config_path PARENT_PATH config_dir)
  message(STATUS "${name} root determined to be: ${root}")
  message(STATUS "${name} package config directory determined to be: ${config_dir}")

  set(${name}_ROOT
      "${root}"
      PARENT_SCOPE)
  set(${name}_DIR
      "${config_dir}"
      PARENT_SCOPE)
  list(APPEND CMAKE_PREFIX_PATH "${config_dir}")
  set(CMAKE_PREFIX_PATH
      "${CMAKE_PREFIX_PATH}"
      PARENT_SCOPE)
endfunction()

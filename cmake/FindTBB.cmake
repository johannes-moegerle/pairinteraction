# SPDX-FileCopyrightText: 2024 PairInteraction Developers
# SPDX-License-Identifier: LGPL-3.0-or-later

include(FindPackageHandleStandardArgs)

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
if(ONEAPI_PYTHON)
  execute_process(
    COMMAND
      ${ONEAPI_PYTHON} -c "import sys
from importlib.metadata import files, PackageNotFoundError
try:
    tbb_config_path = next(p for p in files('tbb-devel') if 'TBBConfig.cmake' in p.name).locate().resolve()
    tbb_library_path = next(p for p in files('tbb') if 'tbb' in p.stem).locate().resolve()
    print(tbb_library_path.parent.parent, tbb_config_path, sep='|')
except PackageNotFoundError:
    sys.exit(1)"
    RESULT_VARIABLE ONEAPI_RESULT
    OUTPUT_VARIABLE ONEAPI_PATHS
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT ONEAPI_RESULT EQUAL 0)
    message(STATUS "Failed to find the 'tbb-devel' Python package using ${ONEAPI_PYTHON}.")
  else()
    string(REPLACE "|" ";" ONEAPI_PATHS_LIST "${ONEAPI_PATHS}")
    list(GET ONEAPI_PATHS_LIST 0 TBB_ROOT)
    list(GET ONEAPI_PATHS_LIST 1 TBB_CONFIG_FILE)
    cmake_path(SET TBB_ROOT NORMALIZE "${TBB_ROOT}")
    cmake_path(SET TBB_CONFIG_FILE NORMALIZE "${TBB_CONFIG_FILE}")
    get_filename_component(TBB_DIR "${TBB_CONFIG_FILE}" DIRECTORY)
    message(STATUS "TBB root determined to be: ${TBB_ROOT}")
    message(STATUS "TBB package config directory determined to be: ${TBB_DIR}")
    list(APPEND CMAKE_PREFIX_PATH "${TBB_DIR}")
  endif()
else()
  message(STATUS "Python interpreter not found; skip discovering Intel oneAPI libraries.")
endif()

find_package(TBB QUIET CONFIG)

find_package_handle_standard_args(
  TBB CONFIG_MODE
  REASON_FAILURE_MESSAGE
    "TBB is obtained from the 'tbb-devel' Python package. Install the build requirements into the Python environment \
that CMake uses (${ONEAPI_PYTHON}) by running 'pip install -r .build_requirements.txt'.")

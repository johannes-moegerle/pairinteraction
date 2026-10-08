# SPDX-FileCopyrightText: 2024 PairInteraction Developers
# SPDX-License-Identifier: LGPL-3.0-or-later

include(FindPackageHandleStandardArgs)

include("${CMAKE_CURRENT_LIST_DIR}/OneAPIFromPython.cmake")
oneapi_from_python(TBB tbb-devel TBBConfig.cmake tbb tbb)

find_package(TBB QUIET CONFIG)

find_package_handle_standard_args(
  TBB CONFIG_MODE
  REASON_FAILURE_MESSAGE
    "TBB is obtained from the 'tbb-devel' Python package. Install the build requirements into the Python environment \
that CMake uses (${ONEAPI_PYTHON}) by running 'pip install -r .build_requirements.txt'.")

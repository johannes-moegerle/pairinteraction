# SPDX-FileCopyrightText: 2026 PairInteraction Developers
# SPDX-License-Identifier: LGPL-3.0-or-later

include_guard(GLOBAL)

# oneapi_from_python(<name> <devel-package> <config-file> <runtime-package> <library-stem>)
#
# Locate an Intel oneAPI library that is obtained from Python packages. The Python interpreter is asked for the package
# config file <config-file> of the package <devel-package> and for the library whose stem contains <library-stem> in the
# package <runtime-package>. If this succeeds, <name>_ROOT and <name>_DIR are set and the config directory is appended
# to CMAKE_PREFIX_PATH. In any case, ONEAPI_PYTHON is set to the interpreter that was used (empty if none was found) and
# ONEAPI_PYTHON_HINT to advice on how to install the build requirements, for use in failure messages.
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

  if(ONEAPI_PYTHON)
    set(ONEAPI_PYTHON_HINT
        "Install the build requirements into the Python environment that CMake uses (${ONEAPI_PYTHON}) by running \
'pip install -r .build_requirements.txt'.")
  else()
    set(ONEAPI_PYTHON_HINT
        "CMake did not find a Python interpreter. Activate a Python environment, or pass \
-DPython_EXECUTABLE=<path to the python executable> to CMake, and install the build requirements into it by running \
'pip install -r .build_requirements.txt'.")
  endif()
  set(ONEAPI_PYTHON
      "${ONEAPI_PYTHON}"
      PARENT_SCOPE)
  set(ONEAPI_PYTHON_HINT
      "${ONEAPI_PYTHON_HINT}"
      PARENT_SCOPE)

  if(NOT ONEAPI_PYTHON)
    message(STATUS "Python interpreter not found; skip discovering ${name} through the '${devel_package}' package.")
    return()
  endif()

  # Each failure exits with a message on stderr, which is reported below
  execute_process(
    COMMAND
      ${ONEAPI_PYTHON} -c "import sys
from importlib.metadata import files, PackageNotFoundError
def locate(package, matches, description):
    try:
        package_files = files(package)
    except PackageNotFoundError:
        sys.exit(f\"The '{package}' Python package is not installed.\")
    if package_files is None:
        sys.exit(f\"The '{package}' Python package does not list its files.\")
    path = next((p for p in package_files if matches(p)), None)
    if path is None:
        sys.exit(f\"The '{package}' Python package does not contain {description}.\")
    return path.locate().resolve()
devel_package, config_file, runtime_package, library_stem = sys.argv[1:]
config_path = locate(devel_package, lambda p: config_file in p.name, f\"'{config_file}'\")
library_path = locate(runtime_package, lambda p: library_stem in p.stem, f\"a '{library_stem}' library\")
print(library_path.parent.parent, config_path, sep='|')"
      "${devel_package}" "${config_file}" "${runtime_package}" "${library_stem}"
    RESULT_VARIABLE ONEAPI_RESULT
    OUTPUT_VARIABLE ONEAPI_PATHS
    ERROR_VARIABLE ONEAPI_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)

  if(NOT ONEAPI_RESULT EQUAL 0)
    # If the interpreter could not be started, the reason is reported in the result variable instead of on stderr
    if(NOT ONEAPI_ERROR)
      set(ONEAPI_ERROR "${ONEAPI_RESULT}")
    endif()
    message(STATUS "Failed to locate ${name} through the Python packages using ${ONEAPI_PYTHON}: ${ONEAPI_ERROR}")
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

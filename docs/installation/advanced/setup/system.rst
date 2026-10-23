.. _system_setup:

######################################
 Setup of the Development Environment
######################################

This section describes how to set up your system for a :ref:`manual build <manual>` of the C++ backend from source. This
is highly platform dependent, and is described below for each operating system individually. In order to create a Python
environment for the Python backend, refer to our :ref:`Python <python_setup>` setup instructions.

*********
 Windows
*********

In order to be able to compile the source code, you have to install the following tools and dependencies:

Build tools
    - CMake_ for running the build system (at least CMake 3.22 is required)
    - VCPKG_ for managing the C++ packages
    - `Visual Studio`_ or MinGW_ as a compiler. We recommend `Visual Studio`_ and will use it in further descriptions.

You can use VCPKG with :github:`our configuration file <tree/master/vcpkg.json>` to install most C++ dependencies.
`Intel oneAPI MKL`_ and `Intel oneAPI TBB`_ are installed as Python packages together with the :ref:`Python build
dependencies <python_setup>`.

In addition, you need to adjust your path environment variable if you want to use certain tools from the command line.
In order to smoothly run all the commands described on this page, add the following paths to your path environment
variable:

======================= =======================================================
Tool                    Path to add
======================= =======================================================
CMake                   ``C:\path\to\cmake\bin``
VCPKG                   ``C:\path\to\vcpkg``
clang-tidy clang-format ``C:\path\to\VisualStudio\Community\VC\Tools\Llvm\bin``
======================= =======================================================

You can either adjust your environmental variables by using the GUI provided by the operating system, or immediately
from the command line. If you want to use the Windows GUI, which we highly recommend, enter "environment variables" into
the search bar and select "Edit the system environment variables". In the System Properties window, click "Environment
Variables...". If you want to set up the environment for all users, search in the "System variables", otherwise in the
"User variables for UserName" for the variable ``Path``. Click on ``Path`` and then on "Edit". You now see a list of all
paths that have been added to ``Path``. Click on "New" and enter the paths to the tools you want to use in the command
prompt. If you want to only change your environment variables temporarily, you can do so by typing

.. code-block:: bash

    set PATH=%PATH%;C:\your\path\here

.. warning::

    Even though the command ``setx`` allows you to permanently override your path variable from the command line, we
    strongly discourage you from using it, as it might permanently delete parts of your system environment variable.

*******
 macOS
*******

Before compiling the source code, you have to install the following tools and dependencies:

Build tools
    - CMake_ for running the build system (at least CMake 3.22 is required)
    - Homebrew_ and the clang compiler.

You can obtain the C++ dependencies from the :github:`github workflow <tree/master/.github/workflows/cpp-backend.yml>`.

***********
 GNU/Linux
***********

Before compiling the source code, you have to install the following tools and dependencies:

Build tools
    - CMake_ for running the build system (at least CMake 3.22 is required)
    - the distribution's package manager for managing C++ packages
    - gcc or clang compiler for compiling C++

A complete list of all C++ dependencies can be found in the Dockerfiles that we use for continuous integration. The
Dockerfiles are located in the :github:`docker branch <tree/docker/docker>`.

.. _cmake: https://cmake.org/download/

.. _homebrew: https://brew.sh/

.. _intel oneapi mkl: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html

.. _intel oneapi tbb: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onetbb-download.html

.. _mingw: https://www.mingw-w64.org/downloads/

.. _vcpkg: https://github.com/microsoft/vcpkg?tab=readme-ov-file#quick-start-windows

.. _visual studio: https://visualstudio.microsoft.com/downloads/

****************************
 Tips independent of the OS
****************************

.. note::

    `Intel oneAPI TBB`_ is a required dependency. `Intel oneAPI MKL`_ is an optional dependency that provides optimized
    linear algebra routines and the FEAST eigensolver. Without MKL, the build system uses LAPACKE if it is found, and
    otherwise only the Eigen diagonalizer is available. On Linux and Windows, both libraries are installed as Python
    packages (``tbb-devel`` and ``mkl-devel``) together with the Python build dependencies from
    ``.build_requirements.txt``. As long as the Python environment containing these packages is activated, CMake finds
    the libraries automatically and you do not need to set any environment variables. On macOS, MKL is not used and TBB
    is installed via Homebrew, see the :github:`GitHub workflow <tree/master/.github/workflows/cpp-backend.yml>`.

.. note::

    Advanced examples for the usage of CMake to build the software for various operating systems can be found in the
    :github:`workflows <tree/master/.github/workflows>` directory of the PairInteraction repository.

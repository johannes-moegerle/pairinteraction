.. _python_setup:

##############
 Python Setup
##############

In order to install the Python library, it is recommended to create a Python virtual environment where you can install
all dependencies. Therefore, you will first need a package management system to create the Python environment. This way,
you can avoid package conflicts with packages you may need for different software projects. In general, we recommend
using uv_ due to its speed, but as many users are familiar with conda_ or pip_, we also describe their usage. Throughout
the documentation, you can select your package manager in the tabs above the commands.

******************************
 Installing a Package Manager
******************************

.. tabs::

    .. group-tab:: uv

        To install uv_ please follow the instructions on the `uv installation page
        <https://docs.astral.sh/uv/getting-started/installation/>`_. No further requirements (like installing python)
        are needed, as uv_ takes care of everything for you.

    .. group-tab:: conda

        To install conda_ please follow the instructions on the `conda installation page
        <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.

    .. group-tab:: pip

        pip_ is shipped with Python. You only need a Python installation with Python ≥ 3.10, which you can download from
        the `Python website <https://www.python.org/downloads/>`_.

********************************
 Creating a Virtual Environment
********************************

Creating and activating a virtual environment can be done via

.. tabs::

    .. group-tab:: uv

        .. code-block:: bash

            uv venv .venv --python 3.10

            # activate the environment on Linux and macOS
            source .venv/bin/activate

            # activate the environment on Windows
            .venv\Scripts\activate

    .. group-tab:: conda

        .. code-block:: bash

            conda create --name venv python=3.10
            conda activate venv

    .. group-tab:: pip

        .. code-block:: bash

            python3 -m venv .venv

            # activate the environment on Linux and macOS
            source .venv/bin/activate

            # activate the environment on Windows
            .venv\Scripts\activate

Note that you can choose a different Python version by simply changing the version number (with pip_, the version of the
Python interpreter that you use for running ``python3 -m venv`` is used).

*****************************
 Installing the Dependencies
*****************************

In order to install all build dependencies for PairInteraction, run inside the activated environment and the root
directory of the PairInteraction repository

.. tabs::

    .. group-tab:: uv

        .. code-block:: bash

            uv pip install -r .build_requirements.txt

    .. group-tab:: conda

        .. code-block:: bash

            pip install -r .build_requirements.txt

    .. group-tab:: pip

        .. code-block:: bash

            pip install -r .build_requirements.txt

If in addition you want to install all dependencies from the ``pyproject.toml`` file, you can use the following
commands:

.. tabs::

    .. group-tab:: uv

        .. code-block:: bash

            uv pip compile pyproject.toml --group dev > requirements.txt
            uv pip install -r requirements.txt

    .. group-tab:: conda

        .. code-block:: bash

            conda install pip-tools
            pip install --upgrade pip  # installing dependency groups via --group requires pip >= 25.1
            pip-compile pyproject.toml --output-file=requirements.txt
            pip install -r requirements.txt --group dev

    .. group-tab:: pip

        .. code-block:: bash

            pip install --upgrade pip pip-tools  # installing dependency groups via --group requires pip >= 25.1
            pip-compile pyproject.toml --output-file=requirements.txt
            pip install -r requirements.txt --group dev

.. _conda: https://anaconda.org/anaconda/conda

.. _pip: https://pypi.org/project/pip/

.. _uv: https://docs.astral.sh/uv/

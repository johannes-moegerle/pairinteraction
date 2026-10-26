.. _installation:

##############
 Installation
##############

The PairInteraction software runs on Linux, macOS (version 14 or later on Apple Silicon & version 15 or later on Intel),
and Windows. It is compatible with Python ≥ 3.10.

**********************
 Installing from PyPI
**********************

For most users, we recommend installing PairInteraction from the `Python Package Index (PyPI)`_. We strongly advise
performing the installation inside a virtual environment to avoid conflicts with other Python packages. For an easy
setup, even on systems where Python is not yet installed, we recommend using uv_. This blazingly fast package manager is
becoming increasingly popular as an alternative to pip_. Alternatively, you can use conda_ or pip_. Select your package
manager to see the commands for setting up a new virtual environment and installing PairInteraction into it:

.. tabs::

    .. group-tab:: uv

        .. code-block:: bash

            # install the uv package manager on macOS and Linux
            curl -LsSf https://astral.sh/uv/install.sh | sh

            # install the uv package manager on Windows
            powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

            # create a new virtual environment in the current directory
            uv venv --python 3.12

            # activate the environment on macOS and Linux
            source .venv/bin/activate

            # activate the environment on Windows
            .venv\Scripts\activate

            # install PairInteraction
            uv pip install pairinteraction

            # deactivate the environment when you are done using PairInteraction
            deactivate

    .. group-tab:: conda

        .. code-block:: bash

            # create a new conda environment
            conda create --name pairinteraction python=3.12

            # activate the environment
            conda activate pairinteraction

            # install PairInteraction
            pip install pairinteraction

            # deactivate the environment when you are done using PairInteraction
            conda deactivate

    .. group-tab:: pip

        .. code-block:: bash

            # create a new virtual environment in the current directory (requires Python >= 3.10)
            python3 -m venv .venv

            # activate the environment on macOS and Linux
            source .venv/bin/activate

            # activate the environment on Windows
            .venv\Scripts\activate

            # install PairInteraction
            pip install pairinteraction

            # deactivate the environment when you are done using PairInteraction
            deactivate

.. _conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

.. _pip: https://pypi.org/project/pip/

.. _python package index (pypi): https://pypi.org/project/pairinteraction

.. _uv: https://docs.astral.sh/uv/

Next Steps
==========

After the installation, we recommend verifying that everything is working:

.. code-block:: bash

    # print the current version of the PairInteraction library
    pairinteraction --version

    # run a module test to check if everything is working correctly
    pairinteraction test

    # download the databases for the species you want to use (e.g. Rb and Yb174_mqdt)
    pairinteraction database download Rb Yb174_mqdt

If the download command fails, you can also download the databases manually from GitHub and place them in the cache
directory. For more details on this, see the :ref:`database <database>` section of the documentation.

Note that any ``pairinteraction ...`` command can also be run via ``python3 -m pairinteraction ...``.

**Using PairInteraction as a Python Library:** For usage examples of the Python library, visit the :ref:`tutorial-api`
section of the documentation.

**Using the Graphical User Interface (GUI):** The graphical user interface can be started by executing

.. code-block:: bash

    pairinteraction

from the command line. This allows you to do some common and basic calculations without writing any code. For more
information on how to use the graphical user interface, visit the :ref:`tutorial-gui` section of the documentation.

**************************************
 Installing from Flathub (Linux only)
**************************************

If you are on Linux and only interested in the graphical user interface, you can install PairInteraction as a Flatpak_
package. For the installation, you have to first install Flatpak (follow the `setup guide`_ for your Linux distribution)
and then run:

.. code-block:: bash

    flatpak remote-add --if-not-exists flathub https://dl.flathub.org/repo/flathub.flatpakrepo
    flatpak install flathub org.pairinteraction.Pairinteraction

Afterwards, you can start the GUI with:

.. code-block:: bash

    flatpak run org.pairinteraction.Pairinteraction

.. _flatpak: https://flathub.org/apps/org.pairinteraction.Pairinteraction

.. _setup guide: https://flathub.org/setup

**********************
 Building from Source
**********************

For developers and experienced users who want to adjust the source code to their own needs, or want :ref:`to contribute
to the package <getting_started_as_a_contributor>` we recommend building the software from source. Some instructions on
how to do this can be found in the :ref:`advanced`.

.. toctree::
    :hidden:

    advanced/advanced.rst

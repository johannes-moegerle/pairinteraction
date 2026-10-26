.. _documentation:

############################
 Building the Documentation
############################

For building the documentation, we are using Sphinx_. In order to build the documentation, you first need to set up your
:ref:`development environment <system_setup>` and create a :ref:`Python environment <python_setup>`. If not stated
otherwise, all commands described are run from the ``docs`` folder of the PairInteraction repository.

In order to install all dependencies to smoothly run Sphinx_, you should first run from the root directory of the
repository

.. tabs::

    .. group-tab:: uv

        .. code-block:: bash

            # if PairInteraction is already installed via the automatic build
            uv pip install --group docs

            # otherwise, build and install PairInteraction together with the documentation dependencies
            uv pip install -e . --group docs

    .. group-tab:: conda

        .. code-block:: bash

            # if PairInteraction is already installed via the automatic build
            pip install --group docs

            # otherwise, build and install PairInteraction together with the documentation dependencies
            pip install -e . --group docs

    .. group-tab:: pip

        .. code-block:: bash

            # if PairInteraction is already installed via the automatic build
            pip install --group docs

            # otherwise, build and install PairInteraction together with the documentation dependencies
            pip install -e . --group docs

Here, the first command assumes that you have already installed PairInteraction via the :ref:`automatic build
<automatic>`. Note that installing dependency groups via ``--group`` requires pip >= 25.1 if you are using conda or pip.

In order to build the documentation once, you can run

.. code-block:: bash

    make html

You can then open the documentation by opening the file ``docs/_build/html/index.html`` with your browser.

Alternatively, you can let the documentation automatically rebuild by running

.. code-block:: bash

    make livehtml

This will start a local web server that serves the documentation at ``http://127.0.0.1:8000`` and automatically rebuilds
the documentation whenever you change the source code or a documentation file.

.. note::

    On **Windows**, you have to replace the command ``make`` by ``call make.bat``

.. _sphinx: https://www.sphinx-doc.org/en/master/index.html

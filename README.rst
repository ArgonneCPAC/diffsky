diffsky
============

Diffsky is a python library based on JAX for producing mock catalogs based on
`diffstar <https://diffstar.readthedocs.io/en/latest/>`_
and `dsps <https://dsps.readthedocs.io/en/latest/>`_.

Installation
------------
The latest release of diffsky is available for installation with pip or conda-forge::

    $ conda install -c conda-forge diffsky


To install diffsky into your environment from the source code::

    $ cd /path/to/root/diffsky
    $ pip install .


Conda environment
~~~~~~~~~~~~~~~~~
For a typical development environment in conda-forge::

    $ conda create -c conda-forge -n diffsky_env python=3.11 numpy jax pytest ipython jupyter matplotlib scipy h5py diffmah diffstar dsps diffsky


Documentation
-------------

Online documentation for diffsky is available at
`diffsky.readthedocs.io <https://diffsky.readthedocs.io/en/latest/>`_.

Latest version of Diffsky mocks
-------------------------------

For information about specific Diffsky mocks, including data location,
software version compatibilities, intended use-cases and known issues and limitations,
see the `Info about specific Diffsky mocks <https://diffsky.readthedocs.io/en/latest/mocks.html>`_
section of the online documentation.

See `this OpenCosmo tutorial <https://argonnecpac.github.io/opencosmo-examples/demo-diffsky-nersc/>`_
for information about how to access and query Diffsky mocks with OpenCosmo.
See `this tutorial <https://argonnecpac.github.io/opencosmo-examples/demo-diffmah-diffstar/>`_
for a demonstration of how to compute mass accretion and star formation histories.
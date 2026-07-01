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

Latest version of diffsky mocks
-------------------------------

See `this OpenCosmo tutorial <https://github.com/ArgonneCPAC/opencosmo-examples/blob/main/03-Diffsky/demo_diffmah_diffstar.ipynb/>`_
for information about how to access the latest mock with OpenCosmo.
The latest release of the mocks can be downloaded at this URL:

    https://portal.nersc.gov/project/hacc/aphearin/diffsky_data/cosmos_260316_06_13_2026

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

Testing
-------
To run the suite of unit tests::

    $ cd /path/to/root/diffsky
    $ pytest

To build html of test coverage::

    $ pytest -v --cov --cov-report html
    $ open htmlcov/index.html


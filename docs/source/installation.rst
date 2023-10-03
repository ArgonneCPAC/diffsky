Installation instructions
=========================

Dependencies
------------

``diffsky`` requires `numpy <https://numpy.org/>`__ 
and `jax <https://jax.readthedocs.io/en/latest/>`__, 
and also a collection of libraries implementing 
the differentiable modeling ingredients: 
`Diffmah <https://github.com/ArgonneCPAC/diffmah>`_, 
`Diffstar <https://github.com/ArgonneCPAC/diffstar>`_, 
and `DSPS <https://github.com/ArgonneCPAC/dsps>`_.

Installation
------------

For a typical development environment with conda-forge::

    $ conda create -c conda-forge -n diffsky_env python=3.9 numpy jax pytest ipython jupyter matplotlib scipy h5py diffmah diffstar dsps diffsky


Managing dependencies
~~~~~~~~~~~~~~~~~~~~~~~

The above command will create a new environment with all the latest releases
of the Diff+ codes. However, depending on your analysis, 
you may need to install a specific branch of diffsky and/one of its dependencies.
You can do this by cloning the GitHub repo of the code for which you need a custom 
version, checking out the appropriate version, and running::

    $ pip install . --no-deps

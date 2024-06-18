Installation instructions
=========================

To install lsstdesc-diffsky into an existing environment with conda-forge::

    $ conda install -c conda-forge diffsky
    

To create a new environment from scratch::

    $ conda create -c conda-forge -n diffsky_env python=3.11 numpy jax pytest ipython jupyter matplotlib scipy h5py diffmah diffstar dsps diffsky


Managing dependencies
---------------------

The above commands will install all the latest releases of the diffsky dependency chain. 
This includes `numpy <https://numpy.org/>`__ and 
`jax <https://jax.readthedocs.io/en/latest/>`__, 
and also a collection of libraries implementing 
the differentiable modeling ingredients: 
`Diffmah <https://github.com/ArgonneCPAC/diffmah>`_, 
`Diffstar <https://github.com/ArgonneCPAC/diffstar>`_, 
and `DSPS <https://github.com/ArgonneCPAC/dsps>`_.

Depending on your analysis, you may need to install a specific branch of diffsky 
and/one of its dependencies. You can do this by cloning the GitHub repo of the code 
for which you need a custom  version, checking out the appropriate version, 
and running::

    $ pip install . --no-deps

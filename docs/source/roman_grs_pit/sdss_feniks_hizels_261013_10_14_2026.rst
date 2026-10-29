sdss_feniks_hizels_261013_10_14_2026
====================================
This page gives some more info about the ``sdss_feniks_hizels_261013_10_14_2026`` mock.

Disk location
-------------
OpenCosmo-formatted data for this mock can be found here on NERSC:
``/global/cfs/cdirs/hacc/OpenCosmo/LastJourney/synthetic_galaxies/sdss_feniks_hizels_261013_10_14_2026``

Model and simulation specs
--------------------------
- Galaxies in the ``sdss_feniks_hizels_261013_10_14_2026`` mock populate the `Last Journey <https://arxiv.org/abs/2006.01697>`_ cosmological N-body simulation.
    - The mass function in Last Journey simulation is complete to :math:`M_{\mathrm{halo}} \gtrsim 10^{11.6} M_{\odot}`.
    - One Diffsky galaxy is painted onto every resolved subhalo core in the lightcone.
    - In addition to galaxies in resolved subhalo cores, this mock includes synthetic galaxies below the mass resolution limit of the simulation.
        - The variable ``lgmp_min=11.0`` sets the lowest mass synthetic halo included in the mock :math:`10^{\mathrm{lgmp\_min}} M_{\odot} < M_{\mathrm{halo}} < 10^{11.6} M_{\odot}`.
- Mock galaxies span a contiguous sky area of 2578.27 deg\ :sup:`2` across :math:`0 < z < 4`, complete to :math:`i < 25`.
- Columns include emission line strengths for Hα, Hβ, [OII], [OIII], and [SIII], plus photometry in all Roman and Rubin bands.
- The diffsky model parameters used to produce this mock were calibrated by Kumail Zaidi to a compilation of optical-NIR photometry from `SDSS <https://ui.adsabs.harvard.edu/abs/2009ApJS..182..543A/abstract>`_, `FENIKS <https://ui.adsabs.harvard.edu/abs/2024ApJ...969...84Z/abstract>`_, and Hα in `HiZELS <https://ui.adsabs.harvard.edu/abs/2013MNRAS.428.1128S/abstract>`_.
    - The loss function in this calibration targeted SDSS ugriz for :math:`0 < z < 0.2`, FENIKS ugrizyJHK for :math:`0.4 < z < 2.5`, and the Hα LF for :math:`0.8 < z < 2.5`.

Documentation and software
---------------------------
This mock should be used with Diffsky version ``v0.3.11`` installed.
Diffsky is an evolving library and later versions of the code may not be
compatible with this mock.
See `diffsky.readthedocs.io v0.3.11 <https://diffsky.readthedocs.io/en/v0.3.11>`_ for
the documentation of the ``v0.3.11`` version of Diffsky.

The metadata of each file in the mock contains information
about the software versions in the diffsky stack.
See the tutorials for information about inspecting the metadata of the mock.

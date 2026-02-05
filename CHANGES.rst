0.3.5 (unreleased)
-------------------


0.3.4 (2026-02-05)
-------------------
- Reduce mock filesize by 10x by omitting redundant dust parameters (https://github.com/ArgonneCPAC/diffsky/pull/317)
- Add feature to make mocks with/out disk/bulge/knot decomposition (https://github.com/ArgonneCPAC/diffsky/pull/317)
- Require Numpy>=2 (https://github.com/ArgonneCPAC/diffsky/pull/331)
- Include emission line fluxes in output mock (https://github.com/ArgonneCPAC/diffsky/pull/335)
- Mock galaxies populating both synthetic and simulated halos are written out to the same directory (https://github.com/ArgonneCPAC/diffsky/pull/336)


0.3.3 (2026-01-06)
-------------------
- Fix bug in elliptical projections
    - https://github.com/ArgonneCPAC/diffsky/pull/297
- Reoptimize disk/bulge shapes
    - https://github.com/ArgonneCPAC/diffsky/pull/301
- Implement automated plot-generating scripts comparing cosmos to different diffsky models
    - https://github.com/ArgonneCPAC/diffsky/pull/291
- Improve bookkeeping in cross-matching jobs
    - https://github.com/ArgonneCPAC/diffsky/pull/293
    - https://github.com/ArgonneCPAC/diffsky/pull/294
    - https://github.com/ArgonneCPAC/diffsky/pull/299
    - https://github.com/ArgonneCPAC/diffsky/pull/300
- Compute photometry in batches when making mocks
    - https://github.com/ArgonneCPAC/diffsky/pull/295
- Adopt astropy units in mock metadata
    - https://github.com/ArgonneCPAC/diffsky/pull/303
- Add peculiar velocities to output mock
    - https://github.com/ArgonneCPAC/diffsky/pull/304


0.3.2 (2025-12-04)
-------------------
- Resolve bug in the treatment of scatter in the SSP errors.
- Component vs composite SEDs now exactly agree
- Improved CI and unit testing
- See (https://github.com/ArgonneCPAC/diffsky/pull/284) for details.


0.3.1 (2025-11-23)
-------------------
- Update diffsky to be compatible with diffstar v1.0.2 (https://github.com/ArgonneCPAC/diffsky/pull/277)
- Add new module to recompute photometry from mock (https://github.com/ArgonneCPAC/diffsky/pull/272)
- Implement mock-production validation test of recomputed photometry (https://github.com/ArgonneCPAC/diffsky/pull/274)
- Add kernel to decompose SED into Disk/bulge/knot components (https://github.com/ArgonneCPAC/diffsky/pull/248)
- Fix memory leak in SED mock production script (https://github.com/ArgonneCPAC/diffsky/pull/244)
- Parallelize mock-production script (https://github.com/ArgonneCPAC/diffsky/pull/267)
- Compute mock photometry in batches (https://github.com/ArgonneCPAC/diffsky/pull/269)
- Add size and orientation of disk and bulge to galaxies in SED mock
    - https://github.com/ArgonneCPAC/diffsky/pull/254
    - https://github.com/ArgonneCPAC/diffsky/pull/265



0.3.0 (2025-10-03)
-------------------
- Update diffsky to be compatible with diffstar v1.0


0.2.5 (2025-10-01)
-------------------
- Minor changes before updating to diffstar v1.0


0.2.4 (2025-09-02)
-------------------
- Release associated with diffskyopt+kdescent calibration by Alan Pearl


0.2.3 (2025-07-30)
-------------------
- Fix bug in lightcone data loader (https://github.com/ArgonneCPAC/diffsky/pull/195)


0.2.2 (2025-07-17)
-------------------
- Change little h convention of Monte Carlo halo generators (https://github.com/ArgonneCPAC/diffsky/pull/184)


0.2.1 (2025-06-30)
-------------------
- Incorporate synthetic lightcone into mock production pipeline (https://github.com/ArgonneCPAC/diffsky/pull/165)
- Add pipeline to generate lightcone mocks (https://github.com/ArgonneCPAC/diffsky/pull/154)
- Improve accuracy of young-star contribution stellar age PDF (https://github.com/ArgonneCPAC/diffsky/pull/151)
- Implement synthetic lightcones to extend resolution limits (https://github.com/ArgonneCPAC/diffsky/pull/143)


0.2.0 (2025-03-24)
-------------------
- Update population-level photometry models of dust and burstiness


0.1.2 (2024-10-25)
-------------------
- Update calls to diffmah v0.6.1 and diffstar v0.3.2


0.1.1 (2023-10-04)
-------------------
- Update calls to dsps v0.3.4


0.1.0 (2023-10-03)
-------------------
- First release. Compatible with diffstar v0.2.1 and dsps v0.3.3.
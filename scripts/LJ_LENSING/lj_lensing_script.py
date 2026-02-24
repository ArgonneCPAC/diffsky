#!/bin/bash/python

import healpy as hp
import numpy as np
import h5py 
import sys

path_maps = '/lcrc/project/cosmo_ai/prlarsen/LJ_lensingmaps/'
path_gals = sys.argv[1]
synthetic = sys.argv[2]
patch_list = sys.argv[3]

patch_list = np.loadtxt(patch_list).astype(int)

step_list_gals = ['121', '124', '127', '131', '134', '137', '141', '144', '148',
                  '151', '155', '159', '163', '167', '171', '176', '180', '184',
                  '189', '194', '198', '203', '208', '213', '219', '224', '230',
                  '235', '241', '247', '253', '259', '266', '272', '279', '286',
                  '293', '300', '307', '315', '323', '331', '338', '347', '355',
                  '365', '373', '382', '392', '401', '411', '421', '432', '442',
                  '453', '464', '475', '487']

step_list_gals = np.array(step_list_gals)[::-1]

if synthetic=='S':
    path_end = '.diffsky_gals.synthetic_halos.hdf5'
else:
    path_end = '.diffsky_gals.hdf5'
    

mag_cols = []
for band in ['g','r','i','z','y','u']:
    for comp in ['','_bulge','_disk','_knots']:
        mag_cols.append('lsst_'+band+comp)
for band in ['F062','F087','F106','F129','F146','F158','F184','F213','Grism_0thOrder','Grism_1stOrder','Prism']:
    for comp in ['','_bulge','_disk','_knots']:
        mag_cols.append('roman_'+band+comp)
flux_cols = ['Halpha','OII','OIII']

step_max = np.array([484, 468, 453, 439, 426, 413, 400, 388, 376, 365, 355,
       344, 334, 324, 315, 306, 297, 288, 280, 272, 264, 256,
       249, 241, 234, 227, 220, 213, 207, 200, 194, 188, 182, 176, 170,
       165, 159, 154, 148, 143, 138, 133, 128, 123, 118, 114, 109, 105,
       101,  96,  92,  88,  84,  80,  77,  73,  69,  66,  62,  59,  56,
        53,  50,  47])

step_min = np.array([500, 484, 468, 453, 439, 426, 413, 400, 388, 376, 365, 355,
       344, 334, 324, 315, 306, 297, 288, 280, 272, 264, 256,
       249, 241, 234, 227, 220, 213, 207, 200, 194, 188, 182, 176, 170,
       165, 159, 154, 148, 143, 138, 133, 128, 123, 118, 114, 109, 105,
       101,  96,  92,  88,  84,  80,  77,  73,  69,  66,  62,  59,  56,
        53,  50])

def step2a(step,zfin,zstart,nsteps):
    aini = 1./(zstart+1.)
    afin = 1./(zfin+1)
    return aini + (afin - aini)/nsteps*(step+1)
def step2z(step, zfin, zstart, nsteps):
    return 1./step2a(step,zfin,zstart,nsteps)-1.

def wrap_angles_ra_dec(ra, dec):
    dec = np.clip(dec, -90.0 + 1e-8, 90.0 - 1e-8)
    ra = np.mod(ra, 360.0)
    ra = np.where((dec < -90.0 + 1e-8) | (dec > 90.0 - 1e-8), 0.0, ra)
    return ra, dec

def fix_wraparound(a, b):
    return (a - b + 180) % 360 - 180

def linear_interp(x, x0, x1, y0, y1):
    """Linearly interpolate to estimate y at x, given two bounding points."""
    if x1 == x0:
        return 0.5 * (y0 + y1)  
    w = (x - x0) / (x1 - x0)
    return (1 - w) * y0 + w * y1

for step in step_list_gals:
    if int(step) in step_max: 
        print(step)
        idx_step = np.argwhere(step_max==int(step))[0][0]
        assert(step_max[idx_step]==int(step))
        hp_maps = {}
        hp_maps['theta'] = hp.read_map(path_maps +'theta_'+str(step_max[idx_step])+'_'+str(step_min[idx_step])+'.fits')
        hp_maps['phi'] = hp.read_map(path_maps +'phi_'+str(step_max[idx_step])+'_'+str(step_min[idx_step])+'.fits')
        hp_maps['kappa'] = hp.read_map(path_maps +'kappa_'+str(step_max[idx_step])+'_'+str(step_min[idx_step])+'.fits')
        hp_maps['shear1'] = hp.read_map(path_maps +'shear1_'+str(step_max[idx_step])+'_'+str(step_min[idx_step])+'.fits')
        hp_maps['shear2'] = hp.read_map(path_maps +'shear2_'+str(step_max[idx_step])+'_'+str(step_min[idx_step])+'.fits')
        hp_maps['ra_obs'], hp_maps['dec_obs'] = hp.pix2ang(8192,np.arange(hp.nside2npix(8192)),lonlat=True)

        for i in patch_list:
            gal_file = h5py.File(path_gals + 'lc_cores-'+step+'.'+str(i)+path_end,'r+')
            ra_in = gal_file['data']['ra_nfw'][:]
            dec_in = gal_file['data']['dec_nfw'][:]
            z_in = gal_file['data']['redshift_true'][:]

            pix_est = hp.ang2pix(8192,ra_in,dec_in,lonlat=True)
            dtheta_est = fix_wraparound( hp_maps['dec_obs'][pix_est], (90.-np.degrees(hp_maps['theta'][pix_est])))
            dphi_est = hp_maps['ra_obs'][pix_est] - np.degrees(hp_maps['phi'][pix_est])


            ra_obs = ra_in + dphi_est
            dec_obs = dec_in + dtheta_est

            ra_obs, dec_obs =  wrap_angles_ra_dec(ra_obs, dec_obs)

            kappa_gals_deflec = hp.pixelfunc.get_interp_val(hp_maps['kappa'], ra_obs, dec_obs, nest=False, lonlat=True)
            s1_gals_deflec = hp.pixelfunc.get_interp_val(hp_maps['shear1'], ra_obs, dec_obs, nest=False, lonlat=True)
            s2_gals_deflec = hp.pixelfunc.get_interp_val(hp_maps['shear2'], ra_obs, dec_obs, nest=False, lonlat=True)

            if "data/ra_obs" in gal_file: 
               del gal_file["data/ra_obs"]
               del gal_file["data/dec_obs"]
               del gal_file["data/kappa"]
               del gal_file["data/shear1"]
               del gal_file["data/shear2"]
               del gal_file["data/magnification"]

            gal_file.create_dataset("data/ra_obs",data=ra_obs)
            gal_file.create_dataset("data/dec_obs",data=dec_obs)
            gal_file.create_dataset("data/kappa",data=kappa_gals_deflec)
            gal_file.create_dataset("data/shear1",data=s1_gals_deflec)
            gal_file.create_dataset("data/shear2",data=s2_gals_deflec)

            deta = (1-kappa_gals_deflec**2)-(s1_gals_deflec**2+s2_gals_deflec**2)
            mag_vals = 1./(deta)
            gal_file.create_dataset("data/magnification",data=mag_vals) 

            # if original then create backup of unlensed data
            if "data/unlensed_magnitudes" not in gal_file:
                gal_file['data'].create_group("unlensed_magnitudes")
            if "data/unlensed_fluxes" not in gal_file:
                gal_file['data'].create_group("unlensed_fluxes")
            for mag in mag_cols:
                str_mag = "data/unlensed_magnitudes/" + mag
                if str_mag not in gal_file:
                    gal_file.create_dataset("data/unlensed_magnitudes/"+mag, data = gal_file["data/"+mag][:])
            for flux in flux_cols:
                str_flux = "data/unlensed_fluxes/" + flux
                if str_flux not in gal_file:
                    gal_file.create_dataset("data/unlensed_fluxes/"+flux, data = gal_file["data/"+flux][:])

            for mag in mag_cols:
                gal_file['data'][mag][:] = gal_file['data']['unlensed_magnitudes'][mag][:] - 2.5*np.log10(np.abs(mag_vals))
            for flux in flux_cols:
                gal_file['data'][flux][:] = gal_file['data']['unlensed_fluxes'][flux][:] * mag_vals


            gal_file.close()
    elif int(step)==487: # 315-> 324
        idx_step = 0 
        hp_maps = {}
        hp_maps['theta'] = hp.read_map(path_maps +'theta_'+str(step_max[idx_step])+'_'+str(step_min[idx_step])+'.fits')
        hp_maps['phi'] = hp.read_map(path_maps +'phi_'+str(step_max[idx_step])+'_'+str(step_min[idx_step])+'.fits')
        hp_maps['kappa'] = hp.read_map(path_maps +'kappa_'+str(step_max[idx_step])+'_'+str(step_min[idx_step])+'.fits')
        hp_maps['shear1'] = hp.read_map(path_maps +'shear1_'+str(step_max[idx_step])+'_'+str(step_min[idx_step])+'.fits')
        hp_maps['shear2'] = hp.read_map(path_maps +'shear2_'+str(step_max[idx_step])+'_'+str(step_min[idx_step])+'.fits')
        hp_maps['ra_obs'], hp_maps['dec_obs'] = hp.pix2ang(8192,np.arange(hp.nside2npix(8192)),lonlat=True)

        for i in patch_list:
            gal_file = h5py.File(path_gals + 'lc_cores-'+step+'.'+str(i)+path_end,'r+')
            ra_in = gal_file['data']['ra_nfw'][:]
            dec_in = gal_file['data']['dec_nfw'][:]
            z_in = gal_file['data']['redshift_true'][:]

            pix_est = hp.ang2pix(8192,ra_in,dec_in,lonlat=True)
            dtheta_est = fix_wraparound( hp_maps['dec_obs'][pix_est], (90.-np.degrees(hp_maps['theta'][pix_est])))
            dphi_est = hp_maps['ra_obs'][pix_est] - np.degrees(hp_maps['phi'][pix_est])


            ra_obs = ra_in + dphi_est
            dec_obs = dec_in + dtheta_est

            ra_obs, dec_obs =  wrap_angles_ra_dec(ra_obs, dec_obs)

            kappa_gals_deflec = hp.pixelfunc.get_interp_val(hp_maps['kappa'], ra_obs, dec_obs, nest=False, lonlat=True)
            s1_gals_deflec = hp.pixelfunc.get_interp_val(hp_maps['shear1'], ra_obs, dec_obs, nest=False, lonlat=True)
            s2_gals_deflec = hp.pixelfunc.get_interp_val(hp_maps['shear2'], ra_obs, dec_obs, nest=False, lonlat=True)

            if "data/ra_obs" in gal_file:
               del gal_file["data/ra_obs"]
               del gal_file["data/dec_obs"]
               del gal_file["data/kappa"]
               del gal_file["data/shear1"]
               del gal_file["data/shear2"]
               del gal_file["data/magnification"]

            gal_file['data']['ra_obs'] = ra_obs[:]#.create_dataset("data/ra_obs",data=ra_val)
            gal_file['data']['dec_obs'] = dec_obs[:]#.create_dataset("data/dec_obs",data=dec_val)
            gal_file['data']['kappa'] = kappa_gals_deflec[:]#.create_dataset("data/kappa",data=kappa_val)
            gal_file['data']['shear1'] = s1_gals_deflec[:]#.create_dataset("data/shear1",data=s1_val)
            gal_file['data']['shear2'] = s2_gals_deflec[:]#.create_dataset("data/shear2",data=s2_val)
            gal_file['data']['magnification'] = np.ones_like(s2_gals_deflec[:]) #NOTE: add here


            # if original then create backup of unlensed data
            if "data/unlensed_magnitudes" not in gal_file:
                gal_file['data'].create_group("unlensed_magnitudes")
            if "data/unlensed_fluxes" not in gal_file:
                gal_file['data'].create_group("unlensed_fluxes")

            for mag in mag_cols:
                str_mag = "data/unlensed_magnitudes/" + mag
                if str_mag not in gal_file:
                    gal_file.create_dataset("data/unlensed_magnitudes/"+mag, data = gal_file["data/"+mag][:])
            for flux in flux_cols:
                str_flux = "data/unlensed_fluxes/" + flux
                if str_flux not in gal_file:
                    gal_file.create_dataset("data/unlensed_fluxes/"+flux, data = gal_file["data/"+flux][:])

            # step 487 don't need to update current magnitudes
            gal_file.close()
    else:
        print(step)
        step_low = step_min[step_min>=int(step)][-1] # for 338 you want 344, 344-388 is positive minimum
        step_high = step_max[step_max<=int(step)][0]
        print(step_low,step_high)
        idx_step_low = np.argwhere(step_max==int(step_low))[0][0]
        idx_step_high = np.argwhere(step_max==int(step_high))[0][0]
        print(idx_step_low, idx_step_high)
        print(step_max[idx_step_low],step_min[idx_step_low])
        print(step_max[idx_step_high], step_min[idx_step_high])
        z_high = step2z(step_max[idx_step_high]-1, 0.0, 200., 500)
        z_low = step2z(step_max[idx_step_low]-1, 0.0, 200., 500)
 

        hp_maps_low = {}
        hp_maps_high = {}

        hp_maps_low['theta'] = hp.read_map(path_maps +'theta_'+str(step_max[idx_step_low])+'_'+str(step_min[idx_step_low])+'.fits')
        hp_maps_high['theta'] = hp.read_map(path_maps +'theta_'+str(step_max[idx_step_high])+'_'+str(step_min[idx_step_high])+'.fits')
        hp_maps_low['phi'] = hp.read_map(path_maps +'phi_'+str(step_max[idx_step_low])+'_'+str(step_min[idx_step_low])+'.fits')
        hp_maps_high['phi'] = hp.read_map(path_maps +'phi_'+str(step_max[idx_step_high])+'_'+str(step_min[idx_step_high])+'.fits')
        hp_maps_high['kappa'] = hp.read_map(path_maps +'kappa_'+str(step_max[idx_step_high])+'_'+str(step_min[idx_step_high])+'.fits')
        hp_maps_high['shear1'] = hp.read_map(path_maps +'shear1_'+str(step_max[idx_step_high])+'_'+str(step_min[idx_step_high])+'.fits')
        hp_maps_high['shear2'] = hp.read_map(path_maps +'shear2_'+str(step_max[idx_step_high])+'_'+str(step_min[idx_step_high])+'.fits')
        hp_maps_low['kappa'] = hp.read_map(path_maps +'kappa_'+str(step_max[idx_step_low])+'_'+str(step_min[idx_step_low])+'.fits')
        hp_maps_low['shear1'] = hp.read_map(path_maps +'shear1_'+str(step_max[idx_step_low])+'_'+str(step_min[idx_step_low])+'.fits')
        hp_maps_low['shear2'] = hp.read_map(path_maps +'shear2_'+str(step_max[idx_step_low])+'_'+str(step_min[idx_step_low])+'.fits')
        hp_maps_low['ra_obs'], hp_maps_low['dec_obs'] = hp.pix2ang(8192,np.arange(hp.nside2npix(8192)),lonlat=True)


        for i in patch_list:
            gal_file = h5py.File(path_gals + 'lc_cores-'+step+'.'+str(i)+path_end,'r+')
            ra_in = gal_file['data']['ra_nfw'][:]
            dec_in = gal_file['data']['dec_nfw'][:]
            z_in = gal_file['data']['redshift_true'][:]

            pix_est = hp.ang2pix(8192,ra_in,dec_in,lonlat=True)
            dtheta_est_low = fix_wraparound( hp_maps_low['dec_obs'][pix_est], (90.-np.degrees(hp_maps_low['theta'][pix_est])))
            dphi_est_low = hp_maps_low['ra_obs'][pix_est] - np.degrees(hp_maps_low['phi'][pix_est])
            dtheta_est_high = fix_wraparound( hp_maps_low['dec_obs'][pix_est], (90.-np.degrees(hp_maps_high['theta'][pix_est])))
            dphi_est_high = hp_maps_low['ra_obs'][pix_est] - np.degrees(hp_maps_high['phi'][pix_est])

            # may need a boundary correction- can also do interpolation here

            ra_obs = ra_in + dphi_est_low
            dec_obs = dec_in + dtheta_est_low
            #ra_obs, dec_obs =  wrap_angles_ra_dec(ra_obs, dec_obs)

            kappa_gals_deflec_low = hp.pixelfunc.get_interp_val(hp_maps_low['kappa'], ra_obs, dec_obs, nest=False, lonlat=True)#*(180./np.pi)
            s1_gals_deflec_low = hp.pixelfunc.get_interp_val(hp_maps_low['shear1'], ra_obs, dec_obs, nest=False, lonlat=True)#*(180./np.pi)
            s2_gals_deflec_low = hp.pixelfunc.get_interp_val(hp_maps_low['shear2'], ra_obs, dec_obs, nest=False, lonlat=True)#*(180./np.pi)
            ra_obs = ra_in + dphi_est_high
            dec_obs = dec_in + dtheta_est_high
            #ra_obs, dec_obs =  wrap_angles_ra_dec(ra_obs, dec_obs)

            kappa_gals_deflec_high = hp.pixelfunc.get_interp_val(hp_maps_high['kappa'], ra_obs, dec_obs, nest=False, lonlat=True)#*(180./np.pi)
            s1_gals_deflec_high = hp.pixelfunc.get_interp_val(hp_maps_high['shear1'], ra_obs, dec_obs, nest=False, lonlat=True)#*(180./np.pi)
            s2_gals_deflec_high = hp.pixelfunc.get_interp_val(hp_maps_high['shear2'], ra_obs, dec_obs, nest=False, lonlat=True)#*(180./np.pi)

            kappa_val = linear_interp(z_in, z_low, z_high, kappa_gals_deflec_low, kappa_gals_deflec_high)
            s1_val =  linear_interp(z_in, z_low, z_high, s1_gals_deflec_low, s1_gals_deflec_high)
            s2_val =  linear_interp(z_in, z_low, z_high, s2_gals_deflec_low, s2_gals_deflec_high)
            ra_val =  linear_interp(z_in, z_low, z_high, ra_in + dphi_est_low, ra_in + dphi_est_high)
            dec_val =  linear_interp(z_in, z_low, z_high, dec_in + dtheta_est_low ,dec_in + dtheta_est_high)
            if "data/ra_obs" in gal_file:
               del gal_file["data/ra_obs"]
               del gal_file["data/dec_obs"]
               del gal_file["data/kappa"]
               del gal_file["data/shear1"]
               del gal_file["data/shear2"]
               del gal_file["data/magnification"]

            # if original then create backup of unlensed data
            if "data/unlensed_magnitudes" not in gal_file:
                gal_file['data'].create_group("unlensed_magnitudes")
            if "data/unlensed_fluxes" not in gal_file:
                gal_file['data'].create_group("unlensed_fluxes")

            for mag in mag_cols:
                str_mag = "data/unlensed_magnitudes/" + mag
                if str_mag not in gal_file:
                    gal_file.create_dataset("data/unlensed_magnitudes/"+mag, data = gal_file["data/"+mag][:])
            for flux in flux_cols:
                str_flux = "data/unlensed_fluxes/" + flux
                if str_flux not in gal_file:
                    gal_file.create_dataset("data/unlensed_fluxes/"+flux, data = gal_file["data/"+flux][:])


            ra_val, dec_val =  wrap_angles_ra_dec(ra_val, dec_val)

            gal_file['data']['ra_obs'] = ra_val[:]#.create_dataset("data/ra_obs",data=ra_val)
            gal_file['data']['dec_obs'] = dec_val[:]#.create_dataset("data/dec_obs",data=dec_val)
            gal_file['data']['kappa'] = kappa_val[:]#.create_dataset("data/kappa",data=kappa_val)
            gal_file['data']['shear1'] = s1_val[:]#.create_dataset("data/shear1",data=s1_val)
            gal_file['data']['shear2'] = s2_val[:]#.create_dataset("data/shear2",data=s2_val)

            deta = (1-kappa_val[:]**2)-(s1_val[:]**2+s2_val[:]**2)
            mag_vals = 1./(deta)
            gal_file["data"]["magnification"] = mag_vals[:] 

            for mag in mag_cols:
                gal_file['data'][mag][:] = gal_file['data']['unlensed_magnitudes'][mag][:] - 2.5*np.log10(np.abs(mag_vals[:]))
            for flux in flux_cols:
                gal_file['data'][flux][:] = gal_file['data']['unlensed_fluxes'][flux][:] * mag_vals

            gal_file.close()

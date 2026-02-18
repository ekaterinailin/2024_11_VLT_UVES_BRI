#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANDES ETC Batch Processing Wrapper
Allows programmatic use with tabular inputs and outputs
"""

import numpy as np
import pandas as pd
import os
from decimal import Decimal
import warnings
warnings.filterwarnings('ignore')

# Import from the original script
from andes_etc_lib import *

# Constants from original script
PLANK_CONSTANT = 6.62607015e-34
LIGHT_SPEED = 299792458.0
DTEL = 38.5
COBS = 0.28
FCAM = 1.5
TBCK = 283
EBCK = 0.20
SAMPLING = 1
RPOW = 100000
PIXBINNED = 1
NDIT = 1

# Spectral data dictionary
spectral_data = {
    'O5V': 1, 'O9V': 2, 'B0V': 3, 'B1V': 4, 'B3V': 5, 'B8V': 6,
    'A0V': 7, 'A2V': 8, 'A3V': 9, 'A5V': 10, 'F0V': 11, 'F2V': 12,
    'F5V': 13, 'F8V': 14, 'G0V': 15, 'G2V': 16, 'G5V': 17, 'G8V': 18,
    'K0V': 19, 'K2V': 20, 'K5V': 21, 'K7V': 22, 'M0V': 23, 'M2V': 24,
    'M4V': 25, 'M5V': 26, '2800K': 27, '2400K': 28, '2100K': 29,
    '1700K': 30, '2700K': 31, '1900K': 32, '1500K': 33, '2000K': 34,
    '2200K': 35, '2300K': 36
}


def calculate_etc_single(spectral_type, magnitude_band, magnitude_value, 
                         airmass, exposure_time, mag_system='VEGA',
                         return_detailed=False):
    """
    Calculate ETC for a single set of parameters.
    
    Parameters:
    -----------
    spectral_type : str
        Spectral type (e.g., 'G2V', 'M0V', etc.)
    magnitude_band : str
        Magnitude band ('U', 'B', 'V', 'R', 'I', 'Z', 'Y', 'J', 'H', 'K')
    magnitude_value : float
        Magnitude value
    airmass : float
        Airmass value (typically 1.0-2.0)
    exposure_time : float
        Exposure time in seconds
    mag_system : str, optional
        Magnitude system ('VEGA' or 'AB'), default is 'VEGA'
    return_detailed : bool, optional
        If True, return detailed results for all bands and orders
        If False, return summary statistics only
    
    Returns:
    --------
    dict or pd.DataFrame
        If return_detailed=False: dictionary with summary statistics
        If return_detailed=True: DataFrame with detailed results per band/order
    """
    
    # Validate inputs
    if spectral_type not in spectral_data:
        raise ValueError(f"Invalid spectral type: {spectral_type}")
    
    valid_bands = ['U', 'B', 'V', 'R', 'I', 'Z', 'Y', 'J', 'H', 'K']
    if magnitude_band not in valid_bands:
        raise ValueError(f"Invalid magnitude band: {magnitude_band}")
    
    st = spectral_type
    specific_magnitude_band = magnitude_band
    specific_magnitude = magnitude_value
    EXPTIME = exposure_time
    
    # Calculate telescope area
    ATEL = calc_telescope_area(DTEL, COBS)
    
    # Calculate flux for spectral type
    flux_obj_UBV, flux_obj_RIZ, flux_obj_YJH, flux_obj_K = calc_flux_spectral_type(
        st, specific_magnitude_band, specific_magnitude)
    
    # Get atmospheric efficiency
    atm_eff_UBV, atm_eff_RIZ, atm_eff_YJH, atm_eff_K = tapas_efficiency(airmass)
    
    # Calculate total efficiencies
    (effs_UBV, effs_RIZ, effs_YJH, effs_K, 
     effs_UBV_tel, effs_RIZ_tel, effs_YJH_tel, effs_K_tel,
     effs_UBV_inst, effs_RIZ_inst, effs_YJH_inst, effs_K_inst,
     effs_UBV2, effs_RIZ2, effs_YJH2, effs_K2,
     wavelength_UBV, wavelength_RIZ, wavelength_YJH, wavelength_K) = calc_total_efficiencies(
        atm_eff_UBV, atm_eff_RIZ, atm_eff_YJH, atm_eff_K)
    
    # Calculate instrument parameters
    DAPE_UBV, DAPE_RIZ, DAPE_YJH, DAPE_K = calculate_instrument_parameter_values(
        wavelength_UBV, wavelength_RIZ, wavelength_YJH, wavelength_K, 'dape')
    DPIX_UBV, DPIX_RIZ, DPIX_YJH, DPIX_K = calculate_instrument_parameter_values(
        wavelength_UBV, wavelength_RIZ, wavelength_YJH, wavelength_K, 'dpix')
    RON_UBV, RON_RIZ, RON_YJH, RON_K = calculate_instrument_parameter_values(
        wavelength_UBV, wavelength_RIZ, wavelength_YJH, wavelength_K, 'ron')
    DARKCUR_UBV, DARKCUR_RIZ, DARKCUR_YJH, DARKCUR_K = calculate_instrument_parameter_values(
        wavelength_UBV, wavelength_RIZ, wavelength_YJH, wavelength_K, 'darkcur')
    bin_size_UBV, bin_size_RIZ, bin_size_YJH, bin_size_K = calculate_instrument_parameter_values(
        wavelength_UBV, wavelength_RIZ, wavelength_YJH, wavelength_K, 'bin_size')
    photon_UBV, photon_RIZ, photon_YJH, photon_K = calculate_instrument_parameter_values(
        wavelength_UBV, wavelength_RIZ, wavelength_YJH, wavelength_K, 'photon')
    
    # Process band parameters
    ANPIX_UBV, PIXAPE_UBV, TOTPIXRE_UBV, NOISEDET_UBV = process_band_parameters(
        DTEL, FCAM, PIXBINNED, NDIT, EXPTIME, DPIX_UBV, DAPE_UBV, RON_UBV, DARKCUR_UBV)
    ANPIX_RIZ, PIXAPE_RIZ, TOTPIXRE_RIZ, NOISEDET_RIZ = process_band_parameters(
        DTEL, FCAM, PIXBINNED, NDIT, EXPTIME, DPIX_RIZ, DAPE_RIZ, RON_RIZ, DARKCUR_RIZ)
    ANPIX_YJH, PIXAPE_YJH, TOTPIXRE_YJH, NOISEDET_YJH = process_band_parameters(
        DTEL, FCAM, PIXBINNED, NDIT, EXPTIME, DPIX_YJH, DAPE_YJH, RON_YJH, DARKCUR_YJH)
    ANPIX_K, PIXAPE_K, TOTPIXRE_K, NOISEDET_K = process_band_parameters(
        DTEL, FCAM, PIXBINNED, NDIT, EXPTIME, DPIX_K, DAPE_K, RON_K, DARKCUR_K)
    
    # Calculate sky background
    sky_background_UBV, sky_background_RIZ, sky_background_YJH, sky_background_K = sky_background(airmass)
    
    # Calculate background flux
    (NBCK_UBV, NOISEBCK_UBV, NBCK_RIZ, NOISEBCK_RIZ, 
     NBCK_YJH, NOISEBCK_YJH, NBCK_K, NOISEBCK_K) = calculate_background_flux(
        RPOW, EBCK, TBCK, ATEL, wavelength_UBV, wavelength_RIZ, wavelength_YJH, wavelength_K,
        effs_UBV, effs_RIZ, effs_YJH, effs_K, DAPE_UBV, DAPE_RIZ, DAPE_YJH, DAPE_K,
        sky_background_UBV, sky_background_RIZ, sky_background_YJH, sky_background_K)
    
    # Calculate mAB
    mAB_UBV, mAB_RIZ, mAB_YJH, mAB_K = calculate_mAB(
        flux_obj_UBV, flux_obj_RIZ, flux_obj_YJH, flux_obj_K,
        photon_UBV, photon_RIZ, photon_YJH, photon_K,
        wavelength_UBV, wavelength_RIZ, wavelength_YJH, wavelength_K)
    
    # Calculate object signal
    nobj_UBV, nobj_RIZ, nobj_YJH, nobj_K = obj_signal(
        ATEL, EXPTIME, RPOW, effs_UBV, effs_RIZ, effs_YJH, effs_K,
        mAB_UBV, mAB_RIZ, mAB_YJH, mAB_K)
    
    # Calculate S/N
    SN_UBV, SN_RIZ, SN_YJH, SN_K = SN_obj(
        nobj_UBV, nobj_RIZ, nobj_YJH, nobj_K,
        NOISEBCK_UBV, NOISEBCK_RIZ, NOISEBCK_YJH, NOISEBCK_K,
        NOISEDET_UBV, NOISEDET_RIZ, NOISEDET_YJH, NOISEDET_K,
        wavelength_UBV, wavelength_RIZ, wavelength_YJH, wavelength_K)
    
    # Calculate central wavelength values
    (wave_central_UBV, wave_central_RIZ, wave_central_YJH, wave_central_K,
     wave_start_UBV, wave_start_RIZ, wave_start_YJH, wave_start_K,
     wave_end_UBV, wave_end_RIZ, wave_end_YJH, wave_end_K,
     sn_central_UBV, order_UBV, order_RIZ, order_YJH, order_K,
     sn_central_RIZ, sn_central_YJH, sn_central_K,
     eff_central_UBV, eff_central_RIZ, eff_central_YJH, eff_central_K,
     eff_central_UBV_tel, eff_central_RIZ_tel, eff_central_YJH_tel, eff_central_K_tel,
     eff_central_UBV_inst, eff_central_RIZ_inst, eff_central_YJH_inst, eff_central_K_inst,
     eff_central_UBV_atm, eff_central_RIZ_atm, eff_central_YJH_atm, eff_central_K_atm,
     nobj_central_UBV, nobj_central_RIZ, nobj_central_YJH, nobj_central_K) = sn_central_wave(
        wavelength_UBV, wavelength_RIZ, wavelength_YJH, wavelength_K,
        effs_UBV_tel, effs_RIZ_tel, effs_YJH_tel, effs_K_tel,
        effs_UBV_inst, effs_RIZ_inst, effs_YJH_inst, effs_K_inst,
        atm_eff_UBV, atm_eff_RIZ, atm_eff_YJH, atm_eff_K,
        nobj_UBV, nobj_RIZ, nobj_YJH, nobj_K,
        SN_UBV, SN_RIZ, SN_YJH, SN_K)
    
    if return_detailed:
        # Return detailed DataFrame with all bands and orders
        results_list = []
        
        # Process UBV band
        for i in range(len(order_UBV)):
            results_list.append({
                'spectral_type': st,
                'magnitude_band': specific_magnitude_band,
                'magnitude': specific_magnitude,
                'airmass': airmass,
                'exposure_time': EXPTIME,
                'band': 'UBV',
                'order': int(order_UBV[i]),
                'wavelength_nm': wave_central_UBV[i],
                'wavelength_start_nm': wave_start_UBV[i] / 10,
                'wavelength_end_nm': wave_end_UBV[i] / 10,
                'eff_telescope': eff_central_UBV_tel[i],
                'eff_instrument': eff_central_UBV_inst[i],
                'eff_atmosphere': eff_central_UBV_atm[i],
                'eff_total': eff_central_UBV[i],
                'object_counts': nobj_central_UBV[i],
                'snr': sn_central_UBV[i]
            })
        
        # Process RIZ band
        for i in range(len(order_RIZ)):
            results_list.append({
                'spectral_type': st,
                'magnitude_band': specific_magnitude_band,
                'magnitude': specific_magnitude,
                'airmass': airmass,
                'exposure_time': EXPTIME,
                'band': 'RIZ',
                'order': int(order_RIZ[i]),
                'wavelength_nm': wave_central_RIZ[i],
                'wavelength_start_nm': wave_start_RIZ[i],
                'wavelength_end_nm': wave_end_RIZ[i],
                'eff_telescope': eff_central_RIZ_tel[i],
                'eff_instrument': eff_central_RIZ_inst[i],
                'eff_atmosphere': eff_central_RIZ_atm[i],
                'eff_total': eff_central_RIZ[i],
                'object_counts': nobj_central_RIZ[i],
                'snr': sn_central_RIZ[i]
            })
        
        # Process YJH band
        for i in range(len(order_YJH)):
            results_list.append({
                'spectral_type': st,
                'magnitude_band': specific_magnitude_band,
                'magnitude': specific_magnitude,
                'airmass': airmass,
                'exposure_time': EXPTIME,
                'band': 'YJH',
                'order': int(order_YJH[i]),
                'wavelength_nm': wave_central_YJH[i],
                'wavelength_start_nm': wave_start_YJH[i],
                'wavelength_end_nm': wave_end_YJH[i],
                'eff_telescope': eff_central_YJH_tel[i],
                'eff_instrument': eff_central_YJH_inst[i],
                'eff_atmosphere': eff_central_YJH_atm[i],
                'eff_total': eff_central_YJH[i],
                'object_counts': nobj_central_YJH[i],
                'snr': sn_central_YJH[i]
            })
        
        # Process K band
        for i in range(len(order_K)):
            results_list.append({
                'spectral_type': st,
                'magnitude_band': specific_magnitude_band,
                'magnitude': specific_magnitude,
                'airmass': airmass,
                'exposure_time': EXPTIME,
                'band': 'K',
                'order': int(order_K[i]),
                'wavelength_nm': wave_central_K[i],
                'wavelength_start_nm': wave_start_K[i],
                'wavelength_end_nm': wave_end_K[i],
                'eff_telescope': eff_central_K_tel[i],
                'eff_instrument': eff_central_K_inst[i],
                'eff_atmosphere': eff_central_K_atm[i],
                'eff_total': eff_central_K[i],
                'object_counts': nobj_central_K[i],
                'snr': sn_central_K[i]
            })
        
        return pd.DataFrame(results_list)
    
    else:
        # Return summary statistics
        return {
            'spectral_type': st,
            'magnitude_band': specific_magnitude_band,
            'magnitude': specific_magnitude,
            'airmass': airmass,
            'exposure_time': EXPTIME,
            # UBV band summary
            'snr_UBV_mean': np.mean(sn_central_UBV),
            'snr_UBV_median': np.median(sn_central_UBV),
            'snr_UBV_max': np.max(sn_central_UBV),
            'snr_UBV_min': np.min(sn_central_UBV),
            # RIZ band summary
            'snr_RIZ_mean': np.mean(sn_central_RIZ),
            'snr_RIZ_median': np.median(sn_central_RIZ),
            'snr_RIZ_max': np.max(sn_central_RIZ),
            'snr_RIZ_min': np.min(sn_central_RIZ),
            # YJH band summary
            'snr_YJH_mean': np.mean(sn_central_YJH),
            'snr_YJH_median': np.median(sn_central_YJH),
            'snr_YJH_max': np.max(sn_central_YJH),
            'snr_YJH_min': np.min(sn_central_YJH),
            # K band summary
            'snr_K_mean': np.mean(sn_central_K),
            'snr_K_median': np.median(sn_central_K),
            'snr_K_max': np.max(sn_central_K),
            'snr_K_min': np.min(sn_central_K),
        }


def calculate_etc_batch(input_df, return_detailed=False):
    """
    Process multiple ETC calculations from a DataFrame.
    
    Parameters:
    -----------
    input_df : pd.DataFrame
        DataFrame with columns:
        - spectral_type: str
        - magnitude_band: str
        - magnitude: float
        - airmass: float
        - exposure_time: float
        - mag_system: str (optional, default 'VEGA')
    
    return_detailed : bool, optional
        If True, return detailed results for all bands and orders
        If False, return summary statistics only
    
    Returns:
    --------
    pd.DataFrame
        Results for all input rows
    """
    
    results = []
    
    for idx, row in input_df.iterrows():
        try:
            mag_system = row.get('mag_system', 'VEGA')
            
            result = calculate_etc_single(
                spectral_type=row['spectral_type'],
                magnitude_band=row['magnitude_band'],
                magnitude_value=row['magnitude'],
                airmass=row['airmass'],
                exposure_time=row['exposure_time'],
                mag_system=mag_system,
                return_detailed=return_detailed
            )
            
            if return_detailed:
                # result is already a DataFrame
                result['input_row_index'] = idx
                results.append(result)
            else:
                # result is a dictionary
                result['input_row_index'] = idx
                results.append(result)
                
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            if not return_detailed:
                # Add error row
                results.append({
                    'input_row_index': idx,
                    'error': str(e)
                })
    
    if return_detailed:
        # Concatenate all DataFrames
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    else:
        # Create DataFrame from list of dictionaries
        return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    # Example 1: Single calculation
    print("=" * 80)
    print("Example 1: Single calculation (summary)")
    print("=" * 80)
    result = calculate_etc_single(
        spectral_type='G2V',
        magnitude_band='V',
        magnitude_value=10.0,
        airmass=1.2,
        exposure_time=3600,
        return_detailed=False
    )
    print(pd.Series(result))
    
    print("\n" + "=" * 80)
    print("Example 2: Single calculation (detailed)")
    print("=" * 80)
    detailed_result = calculate_etc_single(
        spectral_type='G2V',
        magnitude_band='V',
        magnitude_value=10.0,
        airmass=1.2,
        exposure_time=3600,
        return_detailed=True
    )
    print(detailed_result.head(10))
    print(f"\nTotal rows: {len(detailed_result)}")
    
    print("\n" + "=" * 80)
    print("Example 3: Batch calculation")
    print("=" * 80)
    
    # Create sample input DataFrame
    input_data = pd.DataFrame({
        'spectral_type': ['G2V', 'M0V', 'K5V', 'F5V'],
        'magnitude_band': ['V', 'R', 'I', 'V'],
        'magnitude': [10.0, 12.0, 11.5, 9.5],
        'airmass': [1.2, 1.5, 1.3, 1.1],
        'exposure_time': [3600, 1800, 2400, 4000]
    })
    
    print("Input DataFrame:")
    print(input_data)
    
    print("\nProcessing batch...")
    batch_results = calculate_etc_batch(input_data, return_detailed=False)
    print("\nBatch Results (summary):")
    print(batch_results[['spectral_type', 'magnitude', 'airmass', 
                         'snr_UBV_mean', 'snr_RIZ_mean', 'snr_YJH_mean', 'snr_K_mean']])

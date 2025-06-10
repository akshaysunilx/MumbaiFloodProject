#!/usr/bin/env python3
"""
DayWise Prediction Updated - Advanced Multi-Lead Day Weather Prediction System
==============================================================================

This script processes downloaded GFS data and generates rainfall predictions for 3 lead days.
It handles GRIB to NetCDF conversion, data stacking, normalization, and dual-model predictions.

Features:
- GRIB to NetCDF conversion for all weather variables
- APCP accumulation calculations for 24-hour precipitation windows
- PRATE time series processing with 3-hour accumulations
- NetCDF stacking for multi-variable analysis
- Normalization using pre-computed parameters
- Dual model system: CNN + Transfer Learning with threshold-based selection
- Station-specific predictions for 35+ Mumbai weather stations
- Lead-day specific forecast windows
- API integration with Mumbai Flood Prediction System
"""

import os
import datetime as dt
from datetime import datetime, timedelta
import pygrib
import numpy as np
import pandas as pd
import pickle
from netCDF4 import Dataset
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# Import API connection functions
from connections import awsstation, daywiseprediction

# ------------------------------------
# Configuration / Constants
# ------------------------------------

RUN_HOUR_UTC = 0
MODEL_BASE_DIR = os.path.expanduser('~/Desktop/Dont touch/Automation-main')  # Base directory containing LD_1, LD_2, LD_3

# Required files for processing
REQUIRED_FILES = {
    "stations": "stations_coordinates.csv",
    "normalization": "normalization_params.pkl", 
    "thresholds": "thresholds_dict_90percentile.pkl"
}

# ------------------------------------
# Helper Functions
# ------------------------------------

def determine_run_date():
    """Determine the GFS run date and format strings."""
    now_utc = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    run_date = datetime(now_utc.year, now_utc.month, now_utc.day, RUN_HOUR_UTC)
    pdate_str = run_date.strftime("%Y%m%d")
    hour_str = f"{RUN_HOUR_UTC:02d}"
    date_dir = run_date.strftime("%d-%m-%Y")
    return run_date, pdate_str, hour_str, date_dir

def check_required_files():
    """Check if all required files exist."""
    missing_files = []
    for file_type, filename in REQUIRED_FILES.items():
        if not os.path.isfile(filename):
            missing_files.append(f"{file_type}: {filename}")
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files found")
    return True

def convert_grib_to_netcdf(input_grib, output_nc, shortName, description, units):
    """Convert a single GRIB file to NetCDF format."""
    try:
        with pygrib.open(input_grib) as grbs:
            msgs = grbs.select(shortName=shortName)
            if not msgs:
                raise ValueError(f"No message shortName={shortName} in {input_grib}")
            msg = msgs[0]
            data, lats, lons = msg.data()
        
        # Reshape data to 4D: (time, lat, lon, channel)
        data_4d = data[np.newaxis, :, :, np.newaxis]

        with Dataset(output_nc, "w", format="NETCDF4") as nc:
            # Create dimensions
            nc.createDimension("time", 1)
            nc.createDimension("lat", lats.shape[0])
            nc.createDimension("lon", lons.shape[1])
            nc.createDimension("channel", 1)

            # Create variables
            time_var = nc.createVariable("time", "f4", ("time",))
            lat_var = nc.createVariable("lat", "f4", ("lat",))
            lon_var = nc.createVariable("lon", "f4", ("lon",))
            ch_var = nc.createVariable("channel", "i4", ("channel",))
            var = nc.createVariable(shortName, "f4", ("time", "lat", "lon", "channel"))

            # Set attributes
            time_var.units = "hours since forecast start"
            lat_var.units = "degrees_north"
            lon_var.units = "degrees_east"
            ch_var.units = "1"
            var.units = units
            var.description = description

            # Write data
            time_var[:] = [0]
            lat_var[:] = lats[:, 0]
            lon_var[:] = lons[0, :]
            ch_var[:] = [0]
            var[:, :, :, :] = data_4d

        print(f"✅ Converted: {os.path.basename(input_grib)} → {os.path.basename(output_nc)}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to convert {os.path.basename(input_grib)}: {e}")
        return False

def compute_apcp_accumulation(grib_start, grib_end, output_folder, prefix, lead_day, pdate_str, hour_str):
    """Compute 24-hour precipitation accumulation from APCP GRIB files."""
    try:
        if not os.path.exists(grib_start) or not os.path.exists(grib_end):
            print(f"❌ Missing APCP files for {prefix}")
            return False
            
        with pygrib.open(grib_start) as g1, pygrib.open(grib_end) as g2:
            msg1 = g1.select(shortName="tp")[0]  # Total precipitation
            msg2 = g2.select(shortName="tp")[0]
            d1, lats, lons = msg1.data()
            d2, _, _ = msg2.data()
        
        # Compute 24-hour difference
        diff24 = d2 - d1
        
        def write_single_nc(array, lats, lons, out_path, varname, desc, unit):
            """Helper function to write single NetCDF file."""
            arr4d = array[np.newaxis, :, :, np.newaxis]
            with Dataset(out_path, "w", format="NETCDF4") as nc:
                nc.createDimension("time", 1)
                nc.createDimension("lat", lats.shape[0])
                nc.createDimension("lon", lons.shape[1])
                nc.createDimension("channel", 1)

                tvar = nc.createVariable("time", "f4", ("time",))
                latv = nc.createVariable("lat", "f4", ("lat",))
                lonv = nc.createVariable("lon", "f4", ("lon",))
                chv = nc.createVariable("channel", "i4", ("channel",))
                var = nc.createVariable(varname, "f4", ("time", "lat", "lon", "channel"))

                tvar.units = "hours since forecast start"
                latv.units = "degrees_north"
                lonv.units = "degrees_east"
                chv.units = "1"
                var.units = unit
                var.description = desc

                tvar[:] = [0]
                latv[:] = lats[:, 0]
                lonv[:] = lons[0, :]
                chv[:] = [0]
                var[:, :, :, :] = arr4d

        # Create output filenames
        f_start = f"precip_start_{prefix}_{pdate_str}_t{hour_str}.nc"
        f_end = f"precip_end_{prefix}_{pdate_str}_t{hour_str}.nc"
        f_24 = f"precip_24h_{prefix}_{pdate_str}_t{hour_str}.nc"

        out1 = os.path.join(output_folder, f_start)
        out2 = os.path.join(output_folder, f_end)
        out3 = os.path.join(output_folder, f_24)

        # Write NetCDF files
        write_single_nc(d1, lats, lons, out1, varname="precip_start",
                        desc=f"Total precipitation 0–{12*lead_day}h (mm)", unit="mm")
        write_single_nc(d2, lats, lons, out2, varname="precip_end",
                        desc=f"Total precipitation 0–{12*lead_day+24}h (mm)", unit="mm")
        write_single_nc(diff24, lats, lons, out3, varname="precip_24h",
                        desc=f"Precipitation {12*lead_day}–{12*lead_day+24}h (mm)", unit="mm")

        print(f"✅ APCP accumulation computed for {prefix}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to compute APCP for {prefix}: {e}")
        return False

def stack_all_variables(nc_paths, output_path):
    """Stack multiple NetCDF files into a single multi-channel file."""
    try:
        arrays = []
        coords = {}
        
        for key, path in nc_paths.items():
            if not os.path.exists(path):
                print(f"❌ Missing NetCDF file: {key} - {path}")
                return False
                
            with Dataset(path, "r") as ds:
                # Find data variable (excluding coordinate variables)
                varname = [v for v in ds.variables if v not in ("time", "lat", "lon", "channel")][0]
                data = ds.variables[varname][:]
                arrays.append(data)
                
                # Get coordinates from first file
                if not coords:
                    coords["lat"] = ds.variables["lat"][:]
                    coords["lon"] = ds.variables["lon"][:]
                    coords["time"] = ds.variables["time"][:]
        
        # Stack arrays along channel dimension
        stacked = np.concatenate(arrays, axis=3)
        
        # Write stacked NetCDF
        with Dataset(output_path, "w", format="NETCDF4") as nc:
            nc.createDimension("time", coords["time"].shape[0])
            nc.createDimension("lat", coords["lat"].shape[0])
            nc.createDimension("lon", coords["lon"].shape[0])
            nc.createDimension("channel", stacked.shape[3])

            t_var = nc.createVariable("time", "f4", ("time",))
            lat_v = nc.createVariable("lat", "f4", ("lat",))
            lon_v = nc.createVariable("lon", "f4", ("lon",))
            ch_v = nc.createVariable("channel", "i4", ("channel",))
            data_v = nc.createVariable("data", "f4", ("time", "lat", "lon", "channel"))

            t_var[:] = coords["time"]
            lat_v[:] = coords["lat"]
            lon_v[:] = coords["lon"]
            ch_v[:] = np.arange(stacked.shape[3])
            data_v[:, :, :, :] = stacked

            ch_v.long_name = "Channel indices"
            ch_v.description = ", ".join(nc_paths.keys())

        print(f"✅ Stacked NetCDF created: {os.path.basename(output_path)}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to stack variables: {e}")
        return False

def build_precip_dataframe(prate_folder, prate_start, prate_end, run_date, hour_str):
    """Build precipitation DataFrame from PRATE GRIB files."""
    try:
        fh_list = list(range(prate_start, prate_end + 1, 3))
        sub_lat_vals = [19.50, 19.25, 19.00, 18.75]
        sub_lon_vals = [72.50, 72.75, 73.00, 73.25]

        # Create column names
        columns = []
        for fh in fh_list:
            for lat in sub_lat_vals:
                for lon in sub_lon_vals:
                    columns.append(f"Prec_{lat:.2f}_{lon:.2f}_{fh:03d}")

        # Convert to IST
        run_time_local = run_date.replace(tzinfo=dt.timezone.utc) \
                         .astimezone(dt.timezone(timedelta(hours=5, minutes=30))) \
                         .replace(tzinfo=None)
        
        df = pd.DataFrame(index=[run_time_local], columns=columns, dtype=float)

        # Create temporary DataFrame for processing
        date_temp = pd.date_range(start=run_date + timedelta(hours=prate_start),
                                  end=run_date + timedelta(hours=prate_end),
                                  freq="3H")
        temp_df = pd.DataFrame(index=date_temp, columns=range(25), dtype=float)

        lat_full = None
        lon_full = None
        lat_indices = None
        lon_indices = None

        for fh in fh_list:
            grib_name = f"gfs.t{hour_str}z.pgrb2.0p25.f{fh:03d}"
            grib_path = os.path.join(prate_folder, grib_name)
            
            if not os.path.isfile(grib_path):
                print(f"⚠️ Missing PRATE file: {grib_name}")
                continue

            try:
                with pygrib.open(grib_path) as grbs:
                    msgs = grbs.select(name="Precipitation rate")
                    if not msgs:
                        print(f"⚠️ No 'Precipitation rate' in {grib_name}")
                        continue
                    msg = msgs[0]
                    data, lats, lons = msg.data()

                # Initialize coordinates on first successful read
                if lat_indices is None:
                    lat_full = lats[:, 0]
                    lon_full = lons[0, :]
                    target_lats = [19.50, 19.25, 19.00, 18.75, 18.50]
                    target_lons = [72.50, 72.75, 73.00, 73.25, 73.50]
                    lat_indices = [int(np.argmin(np.abs(lat_full - tv))) for tv in target_lats]
                    lon_indices = [int(np.argmin(np.abs(lon_full - tv))) for tv in target_lons]

                # Extract sub-array
                subarray = data[np.ix_(lat_indices, lon_indices)]
                this_time = run_date + timedelta(hours=fh)
                prev_time = this_time - timedelta(hours=3)

                # Convert rates to accumulations
                if (fh % 6) == 0 and prev_time in temp_df.index:
                    temp_df.loc[this_time, 0:25] = (subarray.ravel() * 21600) - temp_df.loc[prev_time, 0:25]
                else:
                    temp_df.loc[this_time, 0:25] = subarray.ravel() * 10800

            except Exception as e:
                print(f"⚠️ Error processing {grib_name}: {e}")
                continue

        # Flatten data for output DataFrame
        flattened = []
        for fh in fh_list:
            t = run_date + timedelta(hours=fh)
            if t in temp_df.index:
                row = temp_df.loc[t].values
            else:
                row = np.full(25, np.nan)
            flattened.extend(row[:16])  # Take first 16 values (4x4 grid)

        df.iloc[0, :] = flattened
        print(f"✅ PRATE DataFrame built: {len([x for x in flattened if not np.isnan(x)])} valid values")
        return df
        
    except Exception as e:
        print(f"❌ Failed to build PRATE DataFrame: {e}")
        return None

def normalize_stack_and_extract(stacked_nc, norm_pickle):
    """Normalize stacked NetCDF data and extract features for model input."""
    try:
        # Read stacked NetCDF
        with Dataset(stacked_nc, "r") as ds:
            varname = [v for v in ds.variables if v not in ("time", "lat", "lon", "channel")][0]
            data = ds.variables[varname][:]
            lats = ds.variables["lat"][:]
            lons = ds.variables["lon"][:]

        # Load normalization parameters
        with open(norm_pickle, "rb") as f:
            norm_params = pickle.load(f)
        means = norm_params["means"]
        stds = norm_params["stds"]

        # Normalize data
        normalized = np.zeros_like(data, dtype=float)
        for i in range(data.shape[3]):
            if stds[i] == 0:
                normalized[..., i] = 0
            else:
                normalized[..., i] = (data[..., i] - means[i]) / stds[i]

        # Extract target coordinates
        target_lats = [18.5, 18.75, 19.0, 19.25]
        target_lons = [72.5, 72.75, 73.0, 73.25]
        lat_idxs = [int(np.argmin(np.abs(lats - tv))) for tv in target_lats]
        lon_idxs = [int(np.argmin(np.abs(lons - tv))) for tv in target_lons]

        # Extract and reshape for model input
        sub = normalized[:, lat_idxs][:, :, lon_idxs]
        model_input = sub.reshape(1, 1, 4, 4, 5)
        
        print(f"✅ Normalized data extracted: shape {model_input.shape}")
        return model_input
        
    except Exception as e:
        print(f"❌ Failed to normalize and extract: {e}")
        return None

def get_station_mapping():
    """Create mapping between station names and station IDs from the API."""
    try:
        aws_stations = awsstation()
        station_mapping = {}
        
        for station_data in aws_stations:
            station_name = station_data['name']
            station_id = station_data['station_id']
            station_mapping[station_name] = station_id
            
        print(f"✅ Retrieved {len(station_mapping)} stations from API")
        return station_mapping
        
    except Exception as e:
        print(f"❌ Failed to get station mapping: {e}")
        return {}

def process_lead_day(lead_day, base_dir, run_date, pdate_str, hour_str, date_dir):
    """Process data and generate predictions for a specific lead day."""
    print(f"\n{'='*60}")
    print(f"PROCESSING LEAD DAY {lead_day}")
    print(f"{'='*60}")
    
    day_dir = os.path.join(base_dir, f"day{lead_day}")
    if not os.path.exists(day_dir):
        print(f"❌ Day directory not found: {day_dir}")
        return False, {}
    
    # Define folder paths
    folders = {
        "APCP": os.path.join(day_dir, "APCP"),
        "PW": os.path.join(day_dir, "PW"),
        "RH": os.path.join(day_dir, "RH"),
        "TEMP_250mb": os.path.join(day_dir, "TEMP_250mb"),
        "PRESSURE_80m": os.path.join(day_dir, "PRESSURE_80m"),
        "PRATE": os.path.join(day_dir, "PRATE"),
        "OUTPUT": os.path.join(day_dir, "OUTPUT")
    }
    
    # Create OUTPUT folder if it doesn't exist
    os.makedirs(folders["OUTPUT"], exist_ok=True)
    
    print(f"\n--- Step 1: Computing APCP Accumulation ---")
    start_fh = 12 * lead_day
    end_fh = start_fh + 24
    grib_start = os.path.join(folders["APCP"], f"gfs.t{hour_str}z.pgrb2.0p25.f{start_fh:03d}")
    grib_end = os.path.join(folders["APCP"], f"gfs.t{hour_str}z.pgrb2.0p25.f{end_fh:03d}")
    prefix = f"day{lead_day}"
    
    if not compute_apcp_accumulation(grib_start, grib_end, folders["APCP"], prefix, lead_day, pdate_str, hour_str):
        print(f"❌ Failed APCP processing for Day {lead_day}")
        return False, {}
    
    print(f"\n--- Step 2: Converting GRIB to NetCDF ---")
    conversions = [
        (os.path.join(folders["PW"], f"gfs.t{hour_str}z.pgrb2.0p25.f036"),
         os.path.join(folders["PW"], f"PWAT_{12*lead_day+12}h_{pdate_str}_t{hour_str}.nc"),
         "pwat", "Precipitable water over entire atmosphere (kg/m^2)", "kg/m^2"),
        
        (os.path.join(folders["RH"], f"gfs.t{hour_str}z.pgrb2.0p25.f036"),
         os.path.join(folders["RH"], f"RH_{12*lead_day+12}h_{pdate_str}_t{hour_str}.nc"),
         "r", "Relative humidity at 500 mb (%)", "%"),
         
        (os.path.join(folders["TEMP_250mb"], f"gfs.t{hour_str}z.pgrb2.0p25.f036"),
         os.path.join(folders["TEMP_250mb"], f"TEMP_250mb_{12*lead_day+12}h_{pdate_str}_t{hour_str}.nc"),
         "t", "Temperature at 250 mb (K)", "K"),
         
        (os.path.join(folders["PRESSURE_80m"], f"gfs.t{hour_str}z.pgrb2.0p25.f036"),
         os.path.join(folders["PRESSURE_80m"], f"PRESSURE_80m_{12*lead_day+12}h_{pdate_str}_t{hour_str}.nc"),
         "pres", "Pressure at 80 m above ground (Pa)", "Pa")
    ]
    
    conversion_success = 0
    for input_grib, output_nc, short_name, description, units in conversions:
        if convert_grib_to_netcdf(input_grib, output_nc, short_name, description, units):
            conversion_success += 1
    
    if conversion_success == 0:
        print(f"❌ No successful GRIB conversions for Day {lead_day}")
        return False, {}
    
    print(f"\n--- Step 3: Stacking Variables ---")
    stacked_fname = f"STACKED_DAY{lead_day}_{pdate_str}_t{hour_str}.nc"
    stacked_output = os.path.join(day_dir, stacked_fname)
    nc_paths = {
        "precip_24h": os.path.join(folders["APCP"], f"precip_24h_{prefix}_{pdate_str}_t{hour_str}.nc"),
        "pwat": os.path.join(folders["PW"], f"PWAT_{12*lead_day+12}h_{pdate_str}_t{hour_str}.nc"),
        "rh": os.path.join(folders["RH"], f"RH_{12*lead_day+12}h_{pdate_str}_t{hour_str}.nc"),
        "temp_250mb": os.path.join(folders["TEMP_250mb"], f"TEMP_250mb_{12*lead_day+12}h_{pdate_str}_t{hour_str}.nc"),
        "pressure_80m": os.path.join(folders["PRESSURE_80m"], f"PRESSURE_80m_{12*lead_day+12}h_{pdate_str}_t{hour_str}.nc")
    }
    
    if not stack_all_variables(nc_paths, stacked_output):
        print(f"❌ Failed to stack variables for Day {lead_day}")
        return False, {}
    
    print(f"\n--- Step 4: Processing PRATE Data ---")
    prate_start = 12 * lead_day + 3
    prate_end = 12 * lead_day + 72
    prate_df = build_precip_dataframe(folders["PRATE"], prate_start, prate_end, run_date, hour_str)
    
    if prate_df is None:
        print(f"❌ Failed to build PRATE DataFrame for Day {lead_day}")
        return False, {}
    
    print(f"\n--- Step 5: Normalizing and Extracting Features ---")
    norm_pickle = os.path.join(os.getcwd(), "normalization_params.pkl")
    extracted_input = normalize_stack_and_extract(stacked_output, norm_pickle)
    
    if extracted_input is None:
        print(f"❌ Failed feature extraction for Day {lead_day}")
        return False, {}
    
    print(f"\n--- Step 6: Running Station Predictions ---")
    stations_csv = os.path.join(os.getcwd(), "stations_coordinates.csv")
    stations_df = pd.read_csv(stations_csv, index_col=0)
    
    thr_pickle = os.path.join(os.getcwd(), "thresholds_dict_90percentile.pkl")
    with open(thr_pickle, "rb") as f:
        thresholds_dict = pickle.load(f)
    
    # Define grid coordinates
    lat_vals = [18.5, 18.75, 19.0, 19.25]
    lon_vals = [72.5, 72.75, 73.0, 73.25]
    lat_lon_arr = np.array([(lat, lon) for lat in lat_vals for lon in lon_vals])
    
    pred_dict_final = {}
    stations_processed = 0
    stations_skipped = 0
    
    for station in stations_df.index:
        try:
            lat_s = stations_df.loc[station, "lat"]
            lon_s = stations_df.loc[station, "lon"]

            # Load lead-day specific models from LD_1, LD_2, or LD_3
            model_dir = os.path.join(MODEL_BASE_DIR, f"LD_{lead_day}")
            cnn_path = os.path.join(model_dir, f"{station}_CNN.h5")
            tl_path = os.path.join(model_dir, f"{station}_tl.h5")
            
            if not os.path.isfile(cnn_path) or not os.path.isfile(tl_path):
                print(f"⚠️ Missing model files for station {station} in LD_{lead_day}")
                stations_skipped += 1
                continue

            # Load models
            cnn_model = load_model(cnn_path)
            tl_model = load_model(tl_path)

            # Find closest grid point
            diffs = lat_lon_arr - np.array([lat_s, lon_s])
            idx = int(np.argmin((diffs**2).sum(axis=1)))
            closest_lat, closest_lon = lat_lon_arr[idx]
            closest_lat_str = f"{closest_lat:.2f}"
            closest_lon_str = f"{closest_lon:.2f}"

            # Extract station-specific PRATE data
            pattern = f"Prec_{closest_lat_str}_{closest_lon_str}_"
            station_cols = [c for c in prate_df.columns if c.startswith(pattern)]
            station_series = prate_df[station_cols]

            # Create daily accumulations
            df_daily = pd.DataFrame(index=station_series.index, columns=[0, 1, 2], dtype=float)
            df_daily[0] = station_series.iloc[:, 0:8].sum(axis=1)
            df_daily[1] = station_series.iloc[:, 8:16].sum(axis=1)
            df_daily[2] = station_series.iloc[:, 16:24].sum(axis=1)

            GFS_vals = df_daily[0].values
            GFS2_vals = df_daily[1].values
            GFS3_vals = df_daily[2].values

            # Run models
            cnn_pred = cnn_model.predict(extracted_input, verbose=0).ravel()
            tl_pred = tl_model.predict(extracted_input, verbose=0).ravel()

            # Get thresholds
            thr_info = thresholds_dict.get(station, {})
            thr_gfs = thr_info.get("GFS", np.inf)
            thr_gfs2 = thr_info.get("GFS2", np.inf)
            thr_gfs3 = thr_info.get("GFS3", np.inf)

            # Apply threshold-based model selection
            final_vals = []
            for i in range(len(GFS_vals)):
                if (GFS_vals[i] > thr_gfs) or (GFS2_vals[i] > thr_gfs2) or (GFS3_vals[i] > thr_gfs3):
                    final_vals.append(tl_pred[i])  # Use Transfer Learning model
                else:
                    final_vals.append(cnn_pred[i])  # Use CNN model

            pred_dict_final[station] = final_vals
            stations_processed += 1
            
            if stations_processed % 5 == 0:
                print(f"   Processed {stations_processed} stations...")

        except Exception as e:
            print(f"⚠️ Error processing station {station}: {e}")
            stations_skipped += 1
            continue

    print(f"\n--- Step 7: Saving Results ---")
    if pred_dict_final:
        final_df = pd.DataFrame(pred_dict_final, index=prate_df.index)
        out_csv = os.path.join(folders["OUTPUT"], f"predicted_values_{date_dir}_t{hour_str}_day{lead_day}.csv")
        final_df.to_csv(out_csv, index=True, float_format="%.2f")
        
        print(f"✅ Predictions saved: {out_csv}")
        print(f"📊 Stations processed: {stations_processed}")
        print(f"⚠️ Stations skipped: {stations_skipped}")
        
        # Show sample predictions
        if len(final_df.columns) > 0:
            sample_stations = list(final_df.columns)[:3]
            print(f"\n📈 Sample predictions for Day {lead_day}:")
            for station in sample_stations:
                pred_val = final_df[station].iloc[0]
                print(f"   {station}: {pred_val:.2f} mm")
        
        return True, pred_dict_final
    else:
        print(f"❌ No predictions generated for Day {lead_day}")
        return False, {}

def send_predictions_to_api(all_predictions, station_mapping):
    """Send predictions to the Mumbai Flood Prediction API."""
    print(f"\n{'='*60}")
    print("SENDING PREDICTIONS TO API")
    print(f"{'='*60}")
    
    stations_sent = 0
    stations_failed = 0
    
    # Get stations that have predictions for all 3 days
    common_stations = set(all_predictions.get('day1', {}).keys())
    for day in ['day2', 'day3']:
        common_stations &= set(all_predictions.get(day, {}).keys())
    
    if not common_stations:
        print("❌ No stations have predictions for all 3 days")
        return
        
    print(f"📡 Sending predictions for {len(common_stations)} stations...")
    
    for station_name in common_stations:
        try:
            # Get station ID from mapping
            station_id = station_mapping.get(station_name)
            if not station_id:
                print(f"⚠️ No station ID found for {station_name}")
                stations_failed += 1
                continue
            
            # Extract predictions for all 3 days
            day1_pred = all_predictions['day1'][station_name][0] if all_predictions['day1'][station_name] else 0.0
            day2_pred = all_predictions['day2'][station_name][0] if all_predictions['day2'][station_name] else 0.0
            day3_pred = all_predictions['day3'][station_name][0] if all_predictions['day3'][station_name] else 0.0
            
            # Prepare API payload
            api_data = {
                'station': station_id,
                'day1': float(day1_pred),
                'day2': float(day2_pred),
                'day3': float(day3_pred)
            }
            
            # Send to API
            response = daywiseprediction(api_data)
            
            if response is not None:
                print(f"✅ {station_name} (ID: {station_id}): Day1={day1_pred:.2f}, Day2={day2_pred:.2f}, Day3={day3_pred:.2f}")
                stations_sent += 1
            else:
                print(f"❌ Failed to send data for {station_name}")
                stations_failed += 1
                
        except Exception as e:
            print(f"❌ Error sending data for {station_name}: {e}")
            stations_failed += 1
            continue
    
    print(f"\n📊 API Transfer Summary:")
    print(f"   ✅ Successfully sent: {stations_sent} stations")
    print(f"   ❌ Failed to send: {stations_failed} stations")
    print(f"   📡 Total API calls: {stations_sent + stations_failed}")

def main():
    """Main function to process all lead days and send to API."""
    print("DayWise Prediction Updated - Advanced Multi-Lead Day Weather Prediction System")
    print("=" * 90)
    
    # Check required files
    if not check_required_files():
        print("\n❌ Cannot proceed without required files")
        print("📋 Required files:")
        for file_type, filename in REQUIRED_FILES.items():
            print(f"   - {filename}")
        return
    
    # Get station mapping from API
    print("\n🔗 Connecting to Mumbai Flood Prediction API...")
    station_mapping = get_station_mapping()
    if not station_mapping:
        print("⚠️ Warning: Could not retrieve station mapping from API")
        print("   Predictions will be saved locally but not sent to API")
    
    # Determine run parameters
    run_date, pdate_str, hour_str, date_dir = determine_run_date()
    
    print(f"\n📅 Processing Date: {run_date.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"📁 Data Directory: {date_dir}")
    print(f"🕐 Forecast Hour: {RUN_HOUR_UTC:02d}Z")
    print(f"🧠 Model Base Directory: {MODEL_BASE_DIR}")
    
    # Check base directory
    base_dir = os.path.join(os.getcwd(), date_dir)
    if not os.path.exists(base_dir):
        print(f"\n❌ Data directory not found: {base_dir}")
        print("💡 Please run gfs_updated.py first to download data")
        return
    
    print(f"📂 Found data directory: {base_dir}")
    
    # Process each lead day
    results = {}
    all_predictions = {}
    
    for lead_day in [1, 2, 3]:
        try:
            print(f"\n🎯 Using models from: {MODEL_BASE_DIR}/LD_{lead_day}/")
            success, predictions = process_lead_day(lead_day, base_dir, run_date, pdate_str, hour_str, date_dir)
            results[f"day{lead_day}"] = "SUCCESS" if success else "FAILED"
            if success and predictions:
                all_predictions[f"day{lead_day}"] = predictions
        except Exception as e:
            print(f"❌ Error processing Lead Day {lead_day}: {e}")
            results[f"day{lead_day}"] = "ERROR"
    
    # Send predictions to API if we have station mapping and predictions
    if station_mapping and all_predictions:
        try:
            send_predictions_to_api(all_predictions, station_mapping)
        except Exception as e:
            print(f"❌ Error sending predictions to API: {e}")
    elif not station_mapping:
        print("\n⚠️ Skipping API upload - no station mapping available")
    elif not all_predictions:
        print("\n⚠️ Skipping API upload - no predictions generated")
    
    # Final summary
    print("\n" + "=" * 90)
    print("FINAL PROCESSING SUMMARY")
    print("=" * 90)
    
    success_count = 0
    for day, status in results.items():
        status_icon = "✅" if status == "SUCCESS" else "❌"
        print(f"{status_icon} {day.upper()}: {status}")
        if status == "SUCCESS":
            success_count += 1
    
    print(f"\n📊 Overall Results: {success_count}/3 lead days processed successfully")
    
    if success_count > 0:
        print(f"\n🎯 Generated Predictions:")
        for day, status in results.items():
            if status == "SUCCESS":
                day_num = day.replace("day", "")
                output_file = os.path.join(base_dir, f"day{day_num}", "OUTPUT", 
                                         f"predicted_values_{date_dir}_t{hour_str}_day{day_num}.csv")
                if os.path.exists(output_file):
                    print(f"   📄 Day {day_num}: {output_file}")
    
    # API integration summary
    if station_mapping and all_predictions:
        print(f"\n🌐 API Integration:")
        print(f"   📡 Mumbai Flood API: https://api.mumbaiflood.in/db/")
        print(f"   🏢 Stations available: {len(station_mapping)}")
        successful_days = [day for day, status in results.items() if status == "SUCCESS"]
        print(f"   📈 Days with predictions: {len(successful_days)}")
    
    if success_count == 0:
        print("\n💡 Troubleshooting Tips:")
        print("   1. Ensure GFS data was downloaded successfully with gfs_updated.py")
        print("   2. Check that all required files exist in the working directory")
        print("   3. Verify model files are available in LD_1, LD_2, LD_3 directories")
        print("   4. Check file permissions and disk space")
        print("   5. Verify API connectivity to https://api.mumbaiflood.in/")
    else:
        print(f"\n🚀 Processing complete! {success_count} prediction files ready for use.")
        if station_mapping and all_predictions:
            print("   📤 Predictions have been sent to the Mumbai Flood Prediction API")

if __name__ == "__main__":
    main()
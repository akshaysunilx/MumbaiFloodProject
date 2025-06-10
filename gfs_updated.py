#!/usr/bin/env python3
"""
GFS Updated Downloader - Advanced Multi-Variable Weather Data Acquisition
=========================================================================

This script downloads comprehensive GFS weather data for 3 lead days with multiple meteorological variables.
Extracts the downloading functionality from the unified workflow for modular use.

For each lead day (1, 2, 3), downloads:
- PRATE: Precipitation rate (f003-f162, every 3 hours) 
- APCP: Accumulated precipitation (lead-day specific windows)
- PWAT: Precipitable water (f036)
- RH: Relative humidity at 500mb (f036)
- TMP: Temperature at 250mb (f036) 
- PRES: Pressure at 80m above ground (f036)

Lead Day Specifications:
- Day 1: APCP f012→f036 (12-36h window)
- Day 2: APCP f024→f048 (24-48h window) 
- Day 3: APCP f036→f060 (36-60h window)
"""

import os
import shutil
import requests
import datetime as dt
from datetime import datetime, timedelta
import urllib3

# Disable SSL warnings for NOAA server
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ------------------------------------
# Configuration / Constants
# ------------------------------------

RUN_HOUR_UTC = 0
BASE_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl"

# Mumbai region boundaries for data subsetting
REGION_BOUNDS = {
    "toplat": 20,
    "bottomlat": 18, 
    "leftlon": 72,
    "rightlon": 74
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

def make_directories(base_dir):
    """Create directory structure for each weather variable."""
    folders = {
        "PRATE": os.path.join(base_dir, "PRATE"),
        "APCP": os.path.join(base_dir, "APCP"),
        "PW": os.path.join(base_dir, "PW"),
        "RH": os.path.join(base_dir, "RH"),
        "TEMP_250mb": os.path.join(base_dir, "TEMP_250mb"),
        "PRESSURE_80m": os.path.join(base_dir, "PRESSURE_80m")
    }
    for path in folders.values():
        os.makedirs(path, exist_ok=True)
    return folders

def download_grib(to_folder, fhours, var_query, level_query, pdate_str, hour_str):
    """
    Download GRIB files from NOAA GFS server.
    
    Parameters:
    -----------
    to_folder : str
        Target directory for downloaded files
    fhours : list
        List of forecast hours (as strings, e.g., ['003', '006'])
    var_query : str 
        Variable query string (e.g., "&var_PRATE=on")
    level_query : str
        Level query string (e.g., "&lev_surface=on")
    pdate_str : str
        Date string in YYYYMMDD format
    hour_str : str
        Hour string in HH format
    """
    success_count = 0
    error_count = 0
    
    for fh in fhours:
        grib_name = f"gfs.t{hour_str}z.pgrb2.0p25.f{fh}"
        
        # Construct download URL with regional subsetting
        url = (
            f"{BASE_URL}?dir=%2Fgfs.{pdate_str}%2F{hour_str}%2Fatmos"
            f"&file={grib_name}"
            f"{var_query}"
            f"{level_query}"
            f"&subregion="
            f"&toplat={REGION_BOUNDS['toplat']}"
            f"&leftlon={REGION_BOUNDS['leftlon']}"
            f"&rightlon={REGION_BOUNDS['rightlon']}"
            f"&bottomlat={REGION_BOUNDS['bottomlat']}"
        )
        
        out_path = os.path.join(to_folder, grib_name)
        
        print(f"[DOWNLOAD] {grib_name} → {to_folder}")
        
        try:
            # Download with streaming to handle large files
            r = requests.get(url, verify=False, stream=True, timeout=60)
            r.raise_for_status()
            r.raw.decode_content = True
            
            with open(out_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)
            
            # Check file size to detect error pages
            file_size = os.path.getsize(out_path)
            if file_size < 1000:  # Likely an error page
                print(f"[WARNING] Small file size ({file_size} bytes) - possible error: {grib_name}")
                error_count += 1
            else:
                print(f"[SUCCESS] {out_path} ({file_size:,} bytes)")
                success_count += 1
                
        except Exception as e:
            print(f"[ERROR] Failed to download {grib_name}: {e}")
            error_count += 1
    
    print(f"[SUMMARY] Downloaded {success_count} files successfully, {error_count} errors")
    return success_count, error_count

def download_lead_day_data(lead_day, base_dir, pdate_str, hour_str):
    """
    Download all required GFS data for a specific lead day.
    
    Parameters:
    -----------
    lead_day : int
        Lead day number (1, 2, or 3)
    base_dir : str
        Base directory for this lead day
    pdate_str : str
        Date string in YYYYMMDD format  
    hour_str : str
        Hour string in HH format
    """
    print(f"\n========== Downloading GFS Data for Lead Day {lead_day} ==========")
    
    # Create directory structure
    day_dir = os.path.join(base_dir, f"day{lead_day}")
    os.makedirs(day_dir, exist_ok=True)
    folders = make_directories(day_dir)
    
    download_stats = {
        "PRATE": (0, 0),
        "APCP": (0, 0), 
        "PW": (0, 0),
        "RH": (0, 0),
        "TEMP_250mb": (0, 0),
        "PRESSURE_80m": (0, 0)
    }
    
    # 1. Download PRATE (Precipitation Rate) - All forecast hours
    print(f"\n--- Downloading PRATE (Precipitation Rate) ---")
    prate_hours = [f"{h:03d}" for h in range(3, 163, 3)]  # f003 to f162, every 3 hours
    download_stats["PRATE"] = download_grib(
        folders["PRATE"], 
        prate_hours, 
        "&var_PRATE=on", 
        "&lev_surface=on",
        pdate_str, 
        hour_str
    )
    
    # 2. Download APCP (Accumulated Precipitation) - Lead day specific
    print(f"\n--- Downloading APCP (Accumulated Precipitation) ---")
    start_fh = 12 * lead_day      # Day 1: 12, Day 2: 24, Day 3: 36
    end_fh = start_fh + 24        # Day 1: 36, Day 2: 48, Day 3: 60
    apcp_hours = [f"{start_fh:03d}", f"{end_fh:03d}"]
    print(f"APCP window for Day {lead_day}: f{start_fh:03d} → f{end_fh:03d}")
    download_stats["APCP"] = download_grib(
        folders["APCP"], 
        apcp_hours, 
        "&var_APCP=on", 
        "&lev_surface=on",
        pdate_str, 
        hour_str
    )
    
    # 3. Download PW (Precipitable Water) - f036 only
    print(f"\n--- Downloading PW (Precipitable Water) ---")
    download_stats["PW"] = download_grib(
        folders["PW"], 
        ["036"], 
        "&var_PWAT=on", 
        "&lev_entire_atmosphere_(considered_as_a_single_layer)=on",
        pdate_str, 
        hour_str
    )
    
    # 4. Download RH (Relative Humidity) - f036 only
    print(f"\n--- Downloading RH (Relative Humidity 500mb) ---")
    download_stats["RH"] = download_grib(
        folders["RH"], 
        ["036"], 
        "&var_RH=on", 
        "&lev_500_mb=on",
        pdate_str, 
        hour_str
    )
    
    # 5. Download TEMP (Temperature 250mb) - f036 only
    print(f"\n--- Downloading TEMP (Temperature 250mb) ---")
    download_stats["TEMP_250mb"] = download_grib(
        folders["TEMP_250mb"], 
        ["036"], 
        "&var_TMP=on", 
        "&lev_250_mb=on",
        pdate_str, 
        hour_str
    )
    
    # 6. Download PRESSURE (Pressure 80m AGL) - f036 only
    print(f"\n--- Downloading PRESSURE (Pressure 80m above ground) ---")
    download_stats["PRESSURE_80m"] = download_grib(
        folders["PRESSURE_80m"], 
        ["036"], 
        "&var_PRES=on", 
        "&lev_80_m_above_ground=on",
        pdate_str, 
        hour_str
    )
    
    # Print summary for this lead day
    print(f"\n========== Lead Day {lead_day} Download Summary ==========")
    total_success = 0
    total_errors = 0
    
    for var_name, (success, errors) in download_stats.items():
        print(f"{var_name:15}: {success:3d} success, {errors:3d} errors")
        total_success += success
        total_errors += errors
    
    print(f"{'TOTAL':15}: {total_success:3d} success, {total_errors:3d} errors")
    print(f"Data saved to: {day_dir}")
    
    return download_stats

def main():
    """Main function to download GFS data for all 3 lead days."""
    print("GFS Updated Downloader - Advanced Multi-Variable Weather Data Acquisition")
    print("=" * 80)
    
    # Determine run parameters
    run_date, pdate_str, hour_str, date_dir = determine_run_date()
    
    print(f"GFS Run Date: {run_date.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Target Directory: {date_dir}")
    print(f"Forecast Hours: {RUN_HOUR_UTC:02d}Z")
    print(f"Region: {REGION_BOUNDS}")
    
    # Create base directory
    base_dir = os.path.join(os.getcwd(), date_dir)
    os.makedirs(base_dir, exist_ok=True)
    
    # Download data for all 3 lead days
    all_stats = {}
    
    for lead_day in [1, 2, 3]:
        try:
            stats = download_lead_day_data(lead_day, base_dir, pdate_str, hour_str)
            all_stats[f"day{lead_day}"] = stats
        except Exception as e:
            print(f"[ERROR] Failed to download data for Lead Day {lead_day}: {e}")
            all_stats[f"day{lead_day}"] = "FAILED"
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL DOWNLOAD SUMMARY")
    print("=" * 80)
    
    grand_total_success = 0
    grand_total_errors = 0
    
    for day, stats in all_stats.items():
        if stats == "FAILED":
            print(f"{day.upper()}: FAILED")
            continue
            
        day_success = sum(s[0] for s in stats.values())
        day_errors = sum(s[1] for s in stats.values())
        grand_total_success += day_success
        grand_total_errors += day_errors
        
        print(f"{day.upper():8}: {day_success:3d} success, {day_errors:3d} errors")
    
    print(f"{'OVERALL':8}: {grand_total_success:3d} success, {grand_total_errors:3d} errors")
    
    if grand_total_errors > 0:
        print(f"\n⚠️  {grand_total_errors} files had download errors (likely due to data availability)")
        print("💡 If files are small (~412 bytes), GFS data may not be ready yet")
        print("🕐 Try again later when 06Z data becomes available (~3 PM IST)")
    
    if grand_total_success > 0:
        print(f"\n✅ Successfully downloaded {grand_total_success} weather files")
        print(f"📁 Data location: {base_dir}")
        print("🚀 Ready for processing with daywiseprediction_updated.py")

if __name__ == "__main__":
    main()
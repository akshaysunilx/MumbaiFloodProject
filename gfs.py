import requests
import shutil
import datetime as dt
from datetime import datetime, timedelta
import os 

def download_gfs_data():
    x = dt.datetime.now().replace(second=0, microsecond=0)
    hour = 6
    day = x.day
    month = x.month
    year = x.year
    pdate = datetime(year, month, day, 11, 00, 00).strftime("%Y%m%d")
    direc = os.path.join('./files', datetime.now().strftime("%d-%m-%Y"))
    direc_prev = os.path.join('./files', (datetime.now() - timedelta(days=1)).strftime("%d-%m-%Y"))
    
    print(f"Downloading GFS data for {pdate} {hour:02d}Z")
    print(f"Target directory: {direc}")
    
    # Delete old files with yesterday's date
    if os.path.exists(direc_prev):
        os.system(f"rm -rf {direc_prev}")
        print(f"Cleaned up: {direc_prev}")
    
    # Create new directory
    os.makedirs(direc, exist_ok=True)
    original_dir = os.getcwd()
    os.chdir(direc)

    base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl"
    forecast_hours = [f"{hour:03d}" for hour in range(15, 163, 3)]  # Changed to match your prediction code

    print(f"Downloading {len(forecast_hours)} files...")
    
    # Iterate over the forecast hours and download the files
    for i, fhour in enumerate(forecast_hours):
        filename = f"gfs.t{str(hour).zfill(2)}z.pgrb2.0p25.f{fhour}"
        url = (f"{base_url}?dir=%2Fgfs.{pdate}%2F{str(hour).zfill(2)}%2Fatmos"
               f"&file={filename}"
               "&var_PRATE=on&var_UGRD=on&var_VGRD=on&lev_10_m_above_ground=on&lev_surface=on"
               "&subregion=&toplat=20&leftlon=72&rightlon=74&bottomlat=18")
        
        print(f"  [{i+1:2d}/{len(forecast_hours)}] {filename}... ", end='')
        
        try:
            r = requests.get(url, verify=False, stream=True, timeout=30)
            r.raw.decode_content = True
            
            with open(filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
            
            # Check file size
            file_size = os.path.getsize(filename)
            if file_size > 1000:
                print(f"✓ ({file_size} bytes)")
            else:
                print(f"✗ (too small: {file_size} bytes)")
                
        except Exception as e:
            print(f"✗ Error: {e}")

    os.chdir(original_dir)
    print(f"Download complete. Files saved to: {direc}")

if __name__ == "__main__":
    download_gfs_data()
from aws_hourly_data import fetch_and_store_hourly_data
import schedule
import time
import csv
import os
from datetime import datetime
from connections import log, systemlog


def save_station_wise_data(data):
    """Save 15-minute rainfall data to individual CSV files for each station"""
    try:
        # Create main folder and station-wise subfolder
        main_folder = os.path.expanduser("~/Desktop/Mumbai_Rainfall_Data")
        stations_folder = os.path.join(main_folder, "station_wise_15min")
        
        # Create folders
        os.makedirs(main_folder, exist_ok=True)
        os.makedirs(stations_folder, exist_ok=True)
        
        # Current timestamp and date
        timestamp = datetime.now().isoformat()
        collection_time = datetime.now().strftime("%H:%M:%S")
        today = datetime.now().strftime("%Y-%m-%d")
        
        stations_processed = 0
        
        # Process each station's data separately
        for station in data:
            try:
                station_id = station.get('station_id', 'unknown')
                station_name = station.get('station_name', 'Unknown Station')
                
                # Create individual CSV file for each station
                # Format: station_01_B_Ward_2025-06-11.csv
                safe_station_name = station_name.replace(' ', '_').replace('/', '_')
                filename = os.path.join(stations_folder, f"station_{station_id:02d}_{safe_station_name}_{today}.csv")
                
                # Check if file exists to determine if we need headers
                file_exists = os.path.exists(filename)
                
                # Open station's CSV file in append mode
                with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
                    fieldnames = [
                        'timestamp', 'collection_time', 'station_id', 'station_name',
                        'rainfall', 'rain_rate', 'temperature', 'humidity', 'wind_speed', 
                        'wind_direction', 'pressure', 'latitude', 'longitude'
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    # Write header if file is new
                    if not file_exists:
                        writer.writeheader()
                        print(f"📋 Created new station file: {filename}")
                    
                    # Write this station's data (exact field mapping from API)
                    writer.writerow({
                        'timestamp': timestamp,
                        'collection_time': collection_time,
                        'station_id': station_id,
                        'station_name': station_name,
                        'rainfall': station.get('rain', 0),  # Current rain amount
                        'rain_rate': station.get('rainRate', 0),  # Rain rate
                        'temperature': station.get('tempOut', ''),  # Outdoor temperature  
                        'humidity': station.get('outHumidity', ''),  # Outdoor humidity
                        'wind_speed': station.get('windSpeed', ''),  # Wind speed
                        'wind_direction': station.get('windDir', ''),  # Wind direction
                        'pressure': station.get('bar', ''),  # Barometric pressure
                        'latitude': station.get('latitude', ''),
                        'longitude': station.get('longitude', '')
                    })
                
                stations_processed += 1
                
            except Exception as e:
                print(f"❌ Error processing station {station.get('station_id', 'unknown')}: {e}")
        
        print(f"✅ Station-wise data saved successfully")
        print(f"📁 Location: {stations_folder}")
        print(f"📊 Stations processed: {stations_processed} | Time: {collection_time}")
        print(f"🗂️ Individual CSV files created/updated for each station")
        
        return stations_processed
        
    except Exception as e:
        print(f"❌ Station-wise save failed: {e}")
        return 0


def collect_station_wise_15min():
    """Main function to collect and save station-wise 15-minute data"""
    try:
        current_time = datetime.now().strftime('%H:%M:%S')
        print(f"\n🕒 Station-wise 15-min collection started at {current_time}")
        print(f"📡 Fetching data from MCGM API...")
        
        # Get rainfall data from MCGM API
        rainfall_data = fetch_and_store_hourly_data()
        
        if rainfall_data:
            # Save to individual station CSV files
            stations_saved = save_station_wise_data(rainfall_data)
            
            if stations_saved > 0:
                log({
                    'log_text': f'Station-wise 15min CSV - {stations_saved} stations saved',
                    'priority': 0
                })
                print(f"🎯 Collection complete: {stations_saved} station files updated")
            else:
                log({
                    'log_text': 'Station-wise 15min CSV Save Failed',
                    'priority': 1
                })
        else:
            print("⚠️ No data received from API")
            log({
                'log_text': 'Station-wise 15min - No Data Received',
                'priority': 1
            })

    except Exception as e:
        error_msg = f'Station-wise 15min Collection Failed: {str(e)}'
        log({
            'log_text': error_msg,
            'priority': 1
        })
        print(f"❌ Error: {e}")


def system_log():
    """System monitoring log"""
    try:
        systemlog({'code': 'Station-wise 15min'}) 
    except Exception as e:
        print(f"System log error: {e}")


# Schedule every 15 minutes
schedule.every(15).minutes.do(collect_station_wise_15min)
schedule.every().minute.do(system_log)

# Display startup info
print("🚀 Starting station-wise 15-minute rainfall monitoring...")
print("📁 Data location: ~/Desktop/Mumbai_Rainfall_Data/station_wise_15min/")
print("🗂️ Creates individual CSV files for each station")
print("⏰ Collection frequency: Every 15 minutes")
print("📊 File format: station_01_B_Ward_2025-06-11.csv")

# Run once immediately on startup
collect_station_wise_15min()

# Main loop
while True:
    schedule.run_pending()
    time.sleep(1)
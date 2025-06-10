import requests
import logging
import json
from connections import awsstation, stationdata

def fetch_aws_data(station_id):
    url = "https://dmwebtwo.mcgm.gov.in/api/tabWeatherForecastData/loadById"
    headers = {
        "Authorization": "Basic RE1BUElVU0VSOkRNYXBpdXNlclBhJCR3b3JkQDEyMzQ="
    }
    payload = {"id": station_id}
    
    print(f"\n🌐 Making API request for station: {station_id}")
    print(f"📡 URL: {url}")
    print(f"📦 Payload: {payload}")
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Raw API Response for station {station_id}:")
        print(f"📊 Status Code: {response.status_code}")
        print(f"🗃️ Full Response Data:")
        print(json.dumps(data, indent=2))
        print(f"📏 Response size: {len(str(data))} characters")
        
        # Show available keys in the response
        if isinstance(data, dict):
            print(f"🔑 Available keys in response: {list(data.keys())}")
        
        parsed_data = parse_data(data)
        print(f"📈 Extracted rainfall value: {parsed_data}")
        print("-" * 60)
        
        return parsed_data
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed for station {station_id}: {e}")
        logging.error(f"Failed to fetch data for station {station_id}: {e}")
        return None

def parse_data(data):
    print(f"🔍 Parsing data...")
    
    # Show what we're looking for
    rainfall_value = data.get('avgRainOneHourAWS')
    print(f"🌧️ Looking for 'avgRainOneHourAWS': {rainfall_value}")
    
    # Let's also see if there are other interesting fields
    if isinstance(data, dict):
        for key, value in data.items():
            if 'rain' in key.lower() or 'precipitation' in key.lower():
                print(f"🌧️ Found rain-related field '{key}': {value}")
    
    return rainfall_value

def parse_value(value):
    if value is None or value == '---':
        return None
    try:
        return float(value)
    except ValueError:
        logging.error(f"Failed to parse value '{value}' as float.")
        return None

def fetch_and_store_hourly_data():
    print("🚀 Starting weather data collection...")
    stations = awsstation()
    print(f"📍 Fetched {len(stations)} stations")
    
    # Show first few stations for reference
    print("\n📋 Station list preview:")
    for i, station in enumerate(stations[:3]):
        print(f"  {i+1}. ID: {station.get('station_id')}, Name: {station.get('name', 'Unknown')}")
    if len(stations) > 3:
        print(f"  ... and {len(stations) - 3} more stations")
    
    print("\n" + "="*60)
    print("Starting data collection from each station...")
    print("="*60)
    
    successful_saves = 0
    failed_saves = 0
    
    for i, station in enumerate(stations):
        print(f"\n📡 Processing station {i+1}/{len(stations)}")
        print(f"🏢 Station ID: {station['station_id']}")
        print(f"📍 Station Name: {station.get('name', 'Unknown')}")
        
        data = fetch_aws_data(station['station_id'])
        
        if data is not None:
            save_station_data(station, data)
            successful_saves += 1
            print(f"✅ Successfully processed station: {station['station_id']}")
        else:
            failed_saves += 1
            print(f"❌ Failed to get data for station: {station['station_id']}")
    
    print(f"\n📊 Summary:")
    print(f"✅ Successful: {successful_saves}")
    print(f"❌ Failed: {failed_saves}")
    print(f"📈 Success Rate: {(successful_saves/(successful_saves+failed_saves)*100):.1f}%")

def save_station_data(station, data):
    payload = {
        'station': station['station_id'],
        'rainfall': data
    }
    
    print(f"\n📤 Preparing to send data to backend:")
    print(f"   🏢 Station: {station['station_id']}")
    print(f"   🌧️ Rainfall: {data}")
    print(f"   📦 Full payload: {payload}")
    
    try:
        response = stationdata(payload)
        print(f"📥 Backend response: {response}")
        
        if response:
            print(f"✅ Data successfully sent to backend")
        else:
            print(f"⚠️ Backend returned empty response")
            
    except Exception as e:
        print(f"❌ Error sending to backend: {e}")

if __name__ == "__main__":
    # Add some startup info
    print("🌦️ Mumbai Weather Data Collection System")
    print("=" * 50)
    
    fetch_and_store_hourly_data()
    
    print("\n🏁 Data collection completed!")
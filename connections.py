import requests
import json

# Monitor = 'http://0.0.0.0:8000/'
BASE_URL = 'https://api.mumbaiflood.in/db/'
Monitor = 'https://monitor.mumbaiflood.in/'


def awsstation():
    url = BASE_URL + 'awsstations/'
    response = requests.get(url)
    stations = response.json()
    
    # Remove duplicates by station_id, keeping the first occurrence
    seen_ids = set()
    unique_stations = []
    
    for station in stations:
        station_id = station['station_id']
        if station_id not in seen_ids:
            unique_stations.append(station)
            seen_ids.add(station_id)
        else:
            print(f"Warning: Skipping duplicate station ID {station_id} for station '{station['name']}'")
    
    print(f"Original stations: {len(stations)}, Unique stations: {len(unique_stations)}")
    return unique_stations

def stationdata(data):
    url = BASE_URL + 'stationdata/'  # or the correct endpoint
    print(data)
    response = requests.post(url, data)
    try:
        return response.json()
    except Exception as e:
        print(f"Error decoding JSON: {e}")
        print(f"Raw response text: {response.text}")
        return None

def awsquaterdata(data):
    url = BASE_URL + 'awsdataforquater/'
    response = requests.post(url, data)
    return response.json()


def fetchstationdata(station):
    url = BASE_URL + 'stationdata/'
    response = requests.get(url, station)
    return response.json()

def daywiseprediction(data):
    url = BASE_URL + 'daywiseprediction/'
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        print(f"Response status code: {response.status_code if 'response' in locals() else 'N/A'}")
        print(f"Response text: {response.text if 'response' in locals() else 'N/A'}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Response text: {response.text if 'response' in locals() else 'N/A'}")
        return None

def hourwiseprediction(data):
    url = BASE_URL + 'hourlyprediction/'
    response = requests.post(url, data)
    return response.json()

def updatetrainstation():
    url = BASE_URL + 'updatetrainstation/'
    response = requests.get(url)
    return response.json()

def tweetdata(data):
    url = BASE_URL + 'tweet/'
    response = requests.post(url, data)
    return response.json()

def log(data):
    try:
        url = Monitor + 'logs/'
        response = requests.post(url, data)
        return response.json()
    except Exception as e:
        print(e)
        return None

def systemlog(data):
    url = Monitor + 'systemlog/'
    response = requests.post(url, data)
    return response.json()
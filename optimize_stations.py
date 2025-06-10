#!/usr/bin/env python3
"""
Station List Optimizer - Remove stations without trained AI models
==================================================================
This script creates an optimized stations file containing only the 35 stations
that have both CNN and Transfer Learning models trained.
"""

import pandas as pd
import os
import shutil
from datetime import datetime

def optimize_station_list():
    """Remove stations without trained models from the stations file"""
    
    print("🎯 Station List Optimizer - Removing Stations Without Models")
    print("=" * 70)
    
    # Configuration
    stations_file = 'stations_coordinates.csv'
    model_dir = '/home/subimal/Desktop/LD_1'
    backup_file = f'stations_coordinates_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    optimized_file = 'stations_coordinates_optimized.csv'
    
    # Step 1: Check if files exist
    if not os.path.exists(stations_file):
        print(f"❌ Error: {stations_file} not found!")
        return False
    
    if not os.path.exists(model_dir):
        print(f"❌ Error: Model directory {model_dir} not found!")
        return False
    
    # Step 2: Load original stations
    print(f"📂 Loading stations from: {stations_file}")
    df_original = pd.read_csv(stations_file, index_col=0)
    print(f"   Original stations: {len(df_original)}")
    
    # Step 3: Create backup
    print(f"💾 Creating backup: {backup_file}")
    shutil.copy2(stations_file, backup_file)
    
    # Step 4: Find stations with both CNN and Transfer Learning models
    print(f"🔍 Scanning for trained models in: {model_dir}")
    available_stations = []
    missing_stations = []
    
    for station in df_original.index:
        cnn_path = os.path.join(model_dir, f'{station}_CNN.h5')
        tl_path = os.path.join(model_dir, f'{station}_tl.h5')
        
        if os.path.exists(cnn_path) and os.path.exists(tl_path):
            available_stations.append(station)
            print(f"   ✅ {station}")
        else:
            missing_stations.append(station)
            print(f"   ❌ {station} (missing models)")
    
    # Step 5: Create optimized dataframe
    df_optimized = df_original.loc[available_stations]
    
    # Step 6: Save optimized file
    print(f"\n💾 Saving optimized stations to: {optimized_file}")
    df_optimized.to_csv(optimized_file)
    
    # Step 7: Replace original file with optimized version
    print(f"🔄 Replacing original file with optimized version")
    shutil.copy2(optimized_file, stations_file)
    
    # Step 8: Summary
    print("\n" + "=" * 70)
    print("📊 OPTIMIZATION SUMMARY")
    print("=" * 70)
    print(f"Original stations:     {len(df_original)}")
    print(f"Stations with models:  {len(available_stations)}")
    print(f"Stations removed:      {len(missing_stations)}")
    print(f"Coverage:              {len(available_stations)/len(df_original)*100:.1f}%")
    
    print(f"\n📁 FILES CREATED:")
    print(f"   🔄 {stations_file} (updated with 35 stations)")
    print(f"   💾 {backup_file} (backup of original)")
    print(f"   📋 {optimized_file} (optimized copy)")
    
    print(f"\n🗑️  REMOVED STATIONS ({len(missing_stations)}):")
    for i, station in enumerate(missing_stations, 1):
        print(f"   {i:2d}. {station}")
    
    print(f"\n✅ REMAINING STATIONS ({len(available_stations)}):")
    for i, station in enumerate(available_stations, 1):
        if i <= 10:  # Show first 10
            print(f"   {i:2d}. {station}")
        elif i == 11:
            print(f"   ... and {len(available_stations)-10} more")
            break
    
    print(f"\n🚀 OPTIMIZATION COMPLETE!")
    print(f"   Your system will now process only the {len(available_stations)} stations with trained models.")
    print(f"   This will eliminate the 18 'missing model' warnings.")
    
    return True

def verify_optimization():
    """Verify the optimization was successful"""
    print("\n🔍 VERIFICATION:")
    
    # Check file sizes
    stations_file = 'stations_coordinates.csv'
    if os.path.exists(stations_file):
        df = pd.read_csv(stations_file, index_col=0)
        print(f"   📋 Current stations file has: {len(df)} stations")
        
        # Check if all stations have models
        model_dir = '/home/subimal/Desktop/LD_1'
        missing_count = 0
        for station in df.index:
            cnn_path = os.path.join(model_dir, f'{station}_CNN.h5')
            tl_path = os.path.join(model_dir, f'{station}_tl.h5')
            if not (os.path.exists(cnn_path) and os.path.exists(tl_path)):
                missing_count += 1
        
        if missing_count == 0:
            print(f"   ✅ All {len(df)} stations have trained models!")
        else:
            print(f"   ⚠️  {missing_count} stations still missing models")
    
if __name__ == "__main__":
    success = optimize_station_list()
    if success:
        verify_optimization()
        print(f"\n🎯 Ready! Run your prediction system again to see the difference!")
    else:
        print(f"\n❌ Optimization failed. Check the error messages above.")

import fastf1
import numpy as np

# SETTINGS (change these to test other races/drivers)
year = 2024
race_name = "Bahrain Grand Prix"
driver_code = "VER"

# Enable cache
fastf1.Cache.enable_cache('cache')

# Load session
session = fastf1.get_session(year, race_name, 'R')
session.load()

# Get driver laps
laps = session.laps.pick_driver(driver_code).reset_index()
stints = laps.groupby('Stint')

# Display real race stint breakdown
print(f"\n🔵 Real Strategy for {driver_code} - {race_name} {year}")
real_total_time = 0

for stint_num, stint_data in stints:
    if stint_data.empty or len(stint_data) < 1:
        continue
    compound = stint_data['Compound'].iloc[0]
    num_laps = len(stint_data)
    lap_times = stint_data['LapTime'].dt.total_seconds()
    stint_time = lap_times.sum()
    real_total_time += stint_time
    print(f"- Stint {stint_num+1}: {compound} for {num_laps} laps (time: {stint_time:.2f}s)")

print(f"▶️ Total Real Race Time: {real_total_time:.2f} seconds")

# Load optimized result for the same driver/race
import pandas as pd
df = pd.read_csv("race_results_summary.csv")

# Find row
match = df[(df['Year'] == year) & (df['Race'] == race_name) & (df['Driver'] == driver_code)]
if match.empty:
    print("\n⚠️ No optimized result found for this driver/race in CSV.")
else:
    print(f"\n🟢 Optimized Strategy for {driver_code} - {race_name} {year}")
    for col in match.columns:
        if "Stint" in col and not pd.isna(match[col].values[0]):
            print(f"- {match[col].values[0]}")
    print(f"▶️ Total Optimized Race Time: {match['Total Time (s)'].values[0]} seconds")

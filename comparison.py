import fastf1
import pandas as pd
import numpy as np

# Enable cache
fastf1.Cache.enable_cache('cache')

targets = [
    (2024, "Bahrain Grand Prix", "VER"),
    (2024, "Bahrain Grand Prix", "HAM"),
    (2024, "Bahrain Grand Prix", "LEC"),
    (2024, "Saudi Arabian Grand Prix", "VER"),
    (2024, "Saudi Arabian Grand Prix", "HAM"),
    (2024, "Saudi Arabian Grand Prix", "LEC"),
    (2024, "Spanish Grand Prix", "VER"),
    (2024, "Spanish Grand Prix", "HAM"),
    (2024, "Spanish Grand Prix", "LEC"),
]

df = pd.read_csv("race_results_summary.csv")
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
results = []

for year, race_name, driver_code in targets:
    print(f"\n============================")
    print(f"🔎 Comparing {driver_code} - {race_name} {year}")

    try:
        session = fastf1.get_session(year, race_name, 'R')
        session.load()

        laps = session.laps.pick_drivers([driver_code]).reset_index()
        stints = laps.groupby('Stint')

        print(f"\n🔵 Real Strategy")
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

        match = df[(df['Year'] == year) & (df['Race'] == race_name) & (df['Driver'] == driver_code)]
        if match.empty:
            print("\n No optimized result found.")
            results.append([year, race_name, driver_code, real_total_time, None, None])
            continue

        print(f"\n Optimized Strategy")
        for col in match.columns:
            if "Stint" in col and not pd.isna(match[col].values[0]):
                print(f"- {match[col].values[0]}")

        optimized_time = float(match['Total Time (s)'].values[0])
        print(f"▶️ Total Optimized Race Time: {optimized_time:.2f} seconds")

        # Comparison
        time_saved = real_total_time - optimized_time
        percent_gain = time_saved / real_total_time * 100
        print(f"\n✅ Time Saved: {time_saved:.2f} seconds ({percent_gain:.2f}%)")

        results.append([year, race_name, driver_code, round(real_total_time, 2), round(optimized_time, 2), round(percent_gain, 2)])

    except Exception as e:
        print(f" Error processing {driver_code} - {race_name} {year}: {str(e)}")
        results.append([year, race_name, driver_code, None, None, None])

# Export results to CSV
summary_df = pd.DataFrame(results, columns=[
    "Year", "Race", "Driver", "Real Time (s)", "Optimized Time (s)", "% Gain"
])
summary_df.to_csv("comparison_summary.csv", index=False)

print("\n All comparisons complete. Results saved to comparison_summary.csv.")

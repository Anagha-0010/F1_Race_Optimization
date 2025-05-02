import fastf1
import numpy as np

def get_race_data(year=2023, race_name='Bahrain Grand Prix', driver='VER'):
    # Enable FastF1 cache
    fastf1.Cache.enable_cache('cache')
    
    # Load session
    session = fastf1.get_session(year, race_name, 'R')
    session.load()


    # Get rain flag from weather data
    rain_flag = session.weather_data['Rainfall'].sum() > 0

    # Get driver laps
    laps = session.laps.pick_drivers(driver).reset_index()
    stints = laps.groupby('Stint')
    stint_list = list(stints)

    tire_compounds = []
    lap_counts = []
    baseline_lap_times = []
    linear_degradation = []
    quadratic_degradation = []
    transitions = []

    for i in range(len(stint_list) - 1):
        prev = stint_list[i][1]
        next = stint_list[i + 1][1]
        if len(prev) < 2 or len(next) < 2:
            continue
        from_comp = prev['Compound'].iloc[0]
        to_comp = next['Compound'].iloc[0]
        pre_avg = prev['LapTime'].dt.total_seconds().iloc[-2:].mean()
        post_avg = next['LapTime'].dt.total_seconds().iloc[:2].mean()
        delta = post_avg - pre_avg
        transitions.append(((from_comp.upper(), to_comp.upper()), delta))

    for stint_num, stint_data in stints:
        if stint_data.empty or len(stint_data) < 5:
            continue
        
        tire = stint_data['Compound'].iloc[0]
        if tire.upper() in ['INTERMEDIATE', 'WET']:
            continue

        lap_times = stint_data['LapTime'].dt.total_seconds()
        laps_idx = np.arange(1, len(lap_times)+1)

        # Early skip for bad data
        if len(laps_idx) < 3 or np.any(np.isnan(lap_times)) or np.all(lap_times == lap_times.iloc[0]):
            print(f"Skipping stint {stint_num} for {driver} — not enough variation or data")
            continue

        try:
            coeffs = np.polyfit(laps_idx, lap_times, 2)
            if np.any(np.isnan(coeffs)):
                print(f"Skipping stint {stint_num} for {driver} — regression returned NaN")
                continue
            baseline = coeffs[2]
            b1 = coeffs[1]
            b2 = coeffs[0]
            if np.isnan(b1) or np.isnan(b2) or np.isnan(baseline):
                print(f"Skipping stint {stint_num} for {driver} — regression returned NaN")
                continue
        except Exception as e:
            print(f"Stint {stint_num} polyfit error for {driver}: {e}")
            continue


        tire_compounds.append(tire)
        lap_counts.append(len(stint_data))
        baseline_lap_times.append(baseline)
        linear_degradation.append(b1)
        quadratic_degradation.append(b2)

    # Map tires
    compound_map = {
        'C1': 'Hard', 'C2': 'Medium', 'C3': 'Soft',
        'C4': 'Soft', 'C5': 'Soft',
        'SOFT': 'Soft', 'MEDIUM': 'Medium', 'HARD': 'Hard'
    }
    mapped_tires = [compound_map.get(t, t) for t in tire_compounds]

    if any(len(lst) == 0 for lst in [tire_compounds, lap_counts, baseline_lap_times]):
        print(f"[get_race_data] Empty result array detected for {driver} in {race_name} {year}")
        return [], [], [], [], [], [], False

    if len(baseline_lap_times) == 0:
        print(f"[get_race_data] No valid stints for {driver} in {race_name} {year}")
        return [], [], [], [], [], [], False
    
    actual_time = laps['LapTime'].dt.total_seconds().sum()


    return np.array(baseline_lap_times), np.array(linear_degradation),np.array(quadratic_degradation), np.array(lap_counts), mapped_tires, transitions, rain_flag, actual_time

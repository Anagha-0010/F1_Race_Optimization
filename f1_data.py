import fastf1
import numpy as np

def get_race_data(year=2024, race_name='Bahrain Grand Prix', driver='VER'):
    # Enable FastF1 cache
    fastf1.Cache.enable_cache('cache')
    
    # Load session
    session = fastf1.get_session(year, race_name, 'R')
    session.load()

    # Get driver laps
    laps = session.laps.pick_driver(driver).reset_index()
    stints = laps.groupby('Stint')

    tire_compounds = []
    lap_counts = []
    baseline_lap_times = []
    degradation_rates = []

    for stint_num, stint_data in stints:
        if stint_data.empty or len(stint_data) < 5:
            continue
        
        tire = stint_data['Compound'].iloc[0]
        lap_times = stint_data['LapTime'].dt.total_seconds()

        baseline = lap_times.iloc[0]
        laps_idx = np.arange(1, len(lap_times)+1)
        coeffs = np.polyfit(laps_idx, lap_times, 1)
        degradation_per_lap = coeffs[0]
        degradation_per_lap = max(0, degradation_per_lap)

        tire_compounds.append(tire)
        lap_counts.append(len(stint_data))
        baseline_lap_times.append(baseline)
        degradation_rates.append(degradation_per_lap)

    # Map tires
    compound_map = {
        'C1': 'Hard', 'C2': 'Medium', 'C3': 'Soft',
        'C4': 'Soft', 'C5': 'Soft',
        'SOFT': 'Soft', 'MEDIUM': 'Medium', 'HARD': 'Hard'
    }
    mapped_tires = [compound_map.get(t, t) for t in tire_compounds]

    return np.array(baseline_lap_times), np.array(degradation_rates), np.array(lap_counts), mapped_tires

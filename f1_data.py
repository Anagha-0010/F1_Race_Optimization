import fastf1
import pandas as pd
import numpy as np

def get_race_data(year, race_name, driver, return_lap_times=False):
    fastf1.Cache.enable_cache('cache')  # Enables local caching

    session = fastf1.get_session(year, race_name, 'R')
    session.load()

    driver_laps = session.laps.pick_drivers([driver]).pick_quicklaps()

    if driver_laps.empty:
        print(f"No lap data found for {driver}")
        return [], [], [], []

    stint_lap_times = []
    a_s, b_s, L_s, stint_tires = [], [], [], []

    current_stint = None
    current_stint_laps = []

    for _, lap in driver_laps.iterlaps():
        stint = lap['Stint']
        compound = lap['Compound']
        lap_time = lap['LapTime'].total_seconds() if pd.notnull(lap['LapTime']) else None

        if lap_time is None:
            continue

        if stint != current_stint:
            if current_stint_laps:
                laps = np.arange(1, len(current_stint_laps) + 1).reshape(-1, 1)
                y = np.array(current_stint_laps)
                X = np.hstack([laps, laps ** 2])
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    a_s.append(float(y[0]))
                    d = float(coeffs[0])
                    b = float(coeffs[1])
                except Exception:
                    d = 0.0
                    b = 0.0
                    a_s.append(float(np.mean(current_stint_laps)))
                b_s.append(max(b, 0))  # ensure convexity
                L_s.append(len(current_stint_laps))
                stint_lap_times.append(current_stint_laps)
                stint_tires.append(current_stint_compound)

            current_stint = stint
            current_stint_compound = compound
            current_stint_laps = [lap_time]
        else:
            current_stint_laps.append(lap_time)

    # Final stint
    if current_stint_laps:
        laps = np.arange(1, len(current_stint_laps) + 1).reshape(-1, 1)
        y = np.array(current_stint_laps)
        X = np.hstack([laps, laps ** 2])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            a_s.append(float(y[0]))
            d = float(coeffs[0])
            b = float(coeffs[1])
        except Exception:
            d = 0.0
            b = 0.0
            a_s.append(float(np.mean(current_stint_laps)))
        b_s.append(max(b, 0))  # ensure convexity
        L_s.append(len(current_stint_laps))
        stint_lap_times.append(current_stint_laps)
        stint_tires.append(current_stint_compound)

    if return_lap_times:
        return a_s, b_s, L_s, stint_tires, stint_lap_times
    else:
        return a_s, b_s, L_s, stint_tires

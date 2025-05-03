import cvxpy as cp
import numpy as np
from f1_data import get_race_data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import csv
import os
import pandas as pd

#finding the tire degradation
def estimate_degradation_params(lap_times, degree=2):
    laps = np.arange(1, len(lap_times) + 1).reshape(-1, 1)
    y = np.array(lap_times)

    if degree == 1:
        model = LinearRegression().fit(laps, y)
        return float(model.intercept_), float(model.coef_[0]), 0.0
    elif degree == 2:
        X_poly = PolynomialFeatures(2, include_bias=False).fit_transform(laps)
        model = LinearRegression().fit(X_poly, y)
        return float(model.intercept_), float(model.coef_[0]), float(model.coef_[1])
    else:
        raise ValueError("Only degree 1 or 2 supported")

# Defining races and drivers
races = [
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

for year, race_name, driver_code in races:
    print(f"\nProcessing {race_name} {year} - Driver: {driver_code}")

    try:
        a_s_raw, b_s_raw, L_s, stint_tires, stint_lap_times = get_race_data(
            year=year, race_name=race_name, driver=driver_code, return_lap_times=True
        )

        M = len(a_s_raw)
        if M == 0:
            print(f"No usable stint data for {driver_code} at {race_name} {year}")
            continue

        N = int(np.sum(L_s))  # Total race laps

        a_s, d_s, b_s = [], [], []
        for laps in stint_lap_times:
            a, d, b = estimate_degradation_params(laps, degree=2)
            a_s.append(a)
            d_s.append(d)
            b_s.append(max(b, 0))  

        f_s = [0.15 if i == 0 else 0.1 if i == 1 else 0.05 for i in range(M)]

        x = cp.Variable(M)
        lap_time_terms = cp.sum(
            cp.multiply(np.array(a_s) + np.array(d_s) + np.array(f_s), x) +
            cp.multiply(np.array(b_s), cp.square(x))
        )
        pit_stop_time = 22
        total_pit_time = (M - 1) * pit_stop_time
        objective = cp.Minimize(lap_time_terms + total_pit_time)

        constraints = [
            cp.sum(x) == N,
            x >= 1,
            x <= L_s
        ]

        U_c = {"Soft": 2, "Medium": 2, "Hard": 2}
        for compound in ['Soft', 'Medium', 'Hard']:
            indices = [i for i, t in enumerate(stint_tires) if t == compound]
            if indices:
                constraints.append(cp.sum([1 for _ in indices]) <= U_c[compound])

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f" Optimization status: {problem.status}")
            continue

        # Rounding
        x_opt = x.value
        x_rounded = np.round(x_opt).astype(int)
        lap_diff = N - np.sum(x_rounded)
        if lap_diff != 0:
            adjust_index = np.argmax(x_opt - np.floor(x_opt)) if lap_diff > 0 else np.argmin(x_opt - np.floor(x_opt))
            x_rounded[adjust_index] += lap_diff

        # Print result
        print(" Optimal Lap Distribution:")
        for i in range(M):
            print(f"- Stint {i+1} ({stint_tires[i]}): {x_rounded[i]} laps")
        print(f"🏁 Total Optimized Race Time: {problem.value:.2f} seconds")


        # Writing to CSV
        csv_file = 'race_results_summary.csv'
        max_stints = 5
        stints = [f"{stint_tires[i]}:{x_rounded[i]}" for i in range(M)]
        stints += [''] * (max_stints - len(stints))
        row_data = [year, race_name, driver_code] + stints + [round(problem.value, 2)]

        header = ['Year', 'Race', 'Driver'] + [f'Stint {i+1}' for i in range(max_stints)] + ['Total Time (s)']

        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip()
            df = df[~((df['Year'] == year) & (df['Race'] == race_name) & (df['Driver'] == driver_code))]
        else:
            df = pd.DataFrame(columns=header)

        new_row = pd.DataFrame([row_data], columns=header)
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(csv_file, index=False)

    except Exception as e:
        print(f" Error processing {driver_code} - {race_name} {year}: {str(e)}")

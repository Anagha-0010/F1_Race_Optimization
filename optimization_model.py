
import cvxpy as cp
import numpy as np
from f1_data import get_race_data
import csv
import os

# Define races and drivers
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
    (2023, "Bahrain Grand Prix", "VER"),
    (2023, "Bahrain Grand Prix", "HAM"),
    (2023, "Bahrain Grand Prix", "LEC"),
    (2023, "Saudi Arabian Grand Prix", "VER"),
    (2023, "Saudi Arabian Grand Prix", "HAM"),
    (2023, "Saudi Arabian Grand Prix", "LEC"),
    (2023, "Spanish Grand Prix", "VER"),
    (2023, "Spanish Grand Prix", "HAM"),
    (2023, "Spanish Grand Prix", "LEC"),
]

# Loop through all races and drivers
for year, race_name, driver_code in races:
    print(f"\\nProcessing {race_name} {year} - Driver: {driver_code}")

    try:
        a_s, b_s, L_s, stint_tires = get_race_data(year=year, race_name=race_name, driver=driver_code)

        M = len(a_s)
        if M == 0:
            print(f"No usable stint data for {driver_code} at {race_name} {year}")
            continue
        N = int(np.sum(L_s))

        x = cp.Variable(M, integer=True)

        lap_time_terms = cp.sum(
            cp.multiply(a_s, x) +
            cp.multiply(b_s/2, cp.square(x)) +
            cp.multiply(b_s/2, x)
        )
        pit_stop_time = 20
        total_pit_time = (M - 1) * pit_stop_time
        objective = cp.Minimize(lap_time_terms + total_pit_time)

        constraints = []
        for i in range(M):
            constraints.append(x[i] <= L_s[i])
        constraints.append(cp.sum(x) == N)

        U_c = {"Soft": 2, "Medium": 2, "Hard": 2}
        for compound in ['Soft', 'Medium', 'Hard']:
            indices = [i for i, t in enumerate(stint_tires) if t == compound]
            constraints.append(cp.sum([1 for _ in indices]) <= U_c[compound])
        constraints.append(x >= 1)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS_BB)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f" Optimization infeasible for {driver_code} at {race_name} {year}")
            continue

        print(f"Optimal Lap Distribution Across Stints (Real Data):")
        for i in range(M):
            print(f"- Stint {i+1} ({stint_tires[i]} Tire): {int(round(x.value[i]))} laps")
        print(f"Total Optimized Race Time: {problem.value:.2f} seconds")

        # Append to log file
        with open('race_results_log.txt', 'a') as f:
            f.write(f"Race: {race_name} {year}\\n")
            f.write(f"Driver: {driver_code}\\n")
            f.write(f"Stint Plan:\\n")
            for i in range(M):
                f.write(f"- Stint {i+1} ({stint_tires[i]} Tire): {int(round(x.value[i]))} laps\\n")
            f.write(f"Total Optimized Race Time: {problem.value:.2f} seconds\\n")
            f.write("-" * 40 + "\\n\\n")

        # Append to CSV file
        csv_file = 'race_results_summary.csv'
        file_exists = os.path.isfile(csv_file)
        stints = [f"{stint_tires[i]}:{int(round(x.value[i]))}" for i in range(M)]

        with open(csv_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                header = ['Year', 'Race', 'Driver'] + [f'Stint {i+1}' for i in range(M)] + ['Total Time (s)']
                writer.writerow(header)
            row = [year, race_name, driver_code] + stints + [round(problem.value, 2)]
            writer.writerow(row)

    except Exception as e:
        print(f"Error processing {race_name} {year} - {driver_code}: {str(e)}")

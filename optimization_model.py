
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

# Define driver style profiles and multipliers
from collections import defaultdict

driver_styles = defaultdict(lambda: 'balanced', {
    'VER': 'aggressive',
    'HAM': 'balanced',
    'LEC': 'conservative',
})

style_multipliers = {
    'aggressive': 1.2,
    'balanced': 1.0,
    'conservative': 0.85
}

# Define rainy races for adjustment
rainy_races = {
    (2024, "Belgian Grand Prix"),
    (2023, "Monaco Grand Prix"),
}



# Loop through all races and drivers
for year, race_name, driver_code in races:
    print(f"\\nProcessing {race_name} {year} - Driver: {driver_code}")

    try:
        a_s, b1_s, b2_s, L_s, stint_tires, transitions, rain_flag = get_race_data(year=year, race_name=race_name, driver=driver_code)

        M = len(a_s)
        if M == 0:
            print(f"No usable stint data for {driver_code} at {race_name} {year}")
            continue
        N = int(np.sum(L_s))

        #Rain Conditions

        if rain_flag:
            print(f"Rain detected for {race_name} {year} — applying adjustments...")
            b1_s = b1_s * 1.5
            b2_s = b2_s * 1.5
            L_s = L_s * 0.8
            #stint_tires = ['Medium' if t == 'Soft' else t for t in stint_tires]



        x = cp.Variable(M, integer=True)

        # Apply degradation multiplier based on driver style
        style = driver_styles[driver_code]
        multiplier = style_multipliers.get(style, 1.0)
        b1_s = b1_s * multiplier
        b2_s = b2_s * multiplier


        # Lap time terms: a_s * x + (b_s / 2) * x^2 + (b_s / 2) * x
        lap_time_terms = cp.sum(
            cp.multiply(a_s, x) +
            cp.multiply(b1_s, x) +
            cp.multiply(b2_s, cp.square(x))
        )

        #dynamic pit stop delays
        rho = cp.Variable(M - 1)
        constraints.append(rho >= 18)
        constraints.append(rho <= 22)
        total_pit_time = cp.sum(rho)

        #adding dynamic fuel load penalty
        mean_lap_time = np.mean(a_s)
        scaling_factor = 0.03  # tunable
        alpha = scaling_factor * mean_lap_time
        fuel_penalty_vector = np.array([alpha * (M - i) for i in range(M)])
        fuel_penalty = cp.sum(cp.multiply(fuel_penalty_vector, x))

        #tire warmup penalty based on tire compounds
        compound_gamma = {'Soft': 1.0, 'Medium': 1.5, 'Hard': 2.0}
        warmup_penalty = sum(
            compound_gamma.get(comp, 1.5) for comp in stint_tires
        )

        #tire compund switch penalty
        USE_CALIBRATED_SWITCH_PENALTIES = False  # Toggle here

        if USE_CALIBRATED_SWITCH_PENALTIES:
            try:    
                penalty_lookup = {}
                for (from_t, to_t), penalty in transitions:
                    penalty_lookup[(from_t, to_t)] = penalty_lookup.get((from_t, to_t), []) + [penalty]
                switch_penalty_matrix = {
                    k: round(np.mean(v), 2) for k, v in penalty_lookup.items()
                }
            except:
                switch_penalty_matrix = {}
        else:
            switch_penalty_matrix = {
                ("Soft", "Medium"): 2.0,
                ("Medium", "Hard"): 2.5,
                ("Hard", "Soft"): 1.5,
                ("Soft", "Soft"): 0.5,
                ("Medium", "Medium"): 0.5,
                ("Hard", "Hard"): 0.5
            }

        compound_switch_penalty = 0
        for k in range(M - 1):
            c1 = stint_tires[k]
            c2 = stint_tires[k + 1]
            compound_switch_penalty += switch_penalty_matrix.get((c1, c2), 1.0)


        # tire compound risk and reliability penalty(complementary to tire switch)
        compound_risk_weights = {'Soft': 0.2, 'Medium': 0.1, 'Hard': 0.0}
        compound_penalty = cp.sum([
            compound_risk_weights.get(stint_tires[i], 0.1) * x[i]
            for i in range(M)
        ])

        # Updated objective with fuel penalty
        objective = cp.Minimize(lap_time_terms + total_pit_time + fuel_penalty +  warmup_penalty + compound_switch_penalty
                                + compound_penalty)


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

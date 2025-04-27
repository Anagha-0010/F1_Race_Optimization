import cvxpy as cp
import numpy as np
from f1_data import get_race_data


# Define races and drivers

races = [
    (2024, "Bahrain Grand Prix", "VER"),
    (2024, "Saudi Arabian Grand Prix", "VER"),
    (2024, "Spanish Grand Prix", "VER"),
    
]


# Loop through all races and drivers

for year, race_name, driver_code in races:
    print(f"\n🔵 Processing {race_name} {year} - Driver: {driver_code}")

    try:
        # Pull race data
        a_s, b_s, L_s, stint_tires = get_race_data(year=year, race_name=race_name, driver=driver_code)
        
        M = len(a_s)
        if M == 0:
            print(f"❌ No usable stint data for {driver_code} at {race_name} {year}")
            continue
        N = int(np.sum(L_s))

        # Optimization Variables
        x = cp.Variable(M, integer=True)

        # Objective Function
        lap_time_terms = cp.sum(
            cp.multiply(a_s, x) +
            cp.multiply(b_s/2, cp.square(x)) +
            cp.multiply(b_s/2, x)
        )
        pit_stop_time = 20  # fixed average pit stop time 
        total_pit_time = (M-1) * pit_stop_time
        objective = cp.Minimize(lap_time_terms + total_pit_time)

        # Constraints
        constraints = []
        for i in range(M):
            constraints.append(x[i] <= L_s[i])
        constraints.append(cp.sum(x) == N)

        for compound in ['Soft', 'Medium', 'Hard']:
            indices = [i for i, t in enumerate(stint_tires) if t == compound]
            constraints.append(cp.sum([1 for _ in indices]) <= 2)  # Max 2 sets per compound

        constraints.append(x >= 1)

        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS_BB)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"❌ Optimization infeasible for {driver_code} at {race_name} {year}")
            continue

        # Output
        print(f"✅ Optimal Lap Distribution Across Stints (Real Data):")
        for i in range(M):
            print(f"- Stint {i+1} ({stint_tires[i]} Tire): {int(round(x.value[i]))} laps")
        print(f"Total Optimized Race Time: {problem.value:.2f} seconds")

        # Save results to file
        with open('race_results_log.txt', 'a') as f:
            f.write(f"Race: {race_name} {year}\n")
            f.write(f"Driver: {driver_code}\n")
            f.write(f"Stint Plan:\n")
            for i in range(M):
                f.write(f"- Stint {i+1} ({stint_tires[i]} Tire): {int(round(x.value[i]))} laps\n")
            f.write(f"Total Optimized Race Time: {problem.value:.2f} seconds\n")
            f.write("-" * 40 + "\n\n")

    except Exception as e:
        print(f"⚠️ Error processing {race_name} {year} - {driver_code}: {str(e)}")

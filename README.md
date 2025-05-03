# F1 Race Strategy Optimization
```
This project models and solves the problem of optimizing race strategies in Formula 1 using convex optimization.
It compares real-world driver performance with a theoretically optimized strategy that minimizes total race time under simplified constraints.
```
# Setup

1. **Clone or download** the repository.

2. Create a local cache folder for FastF1:
   ```bash
   mkdir cache
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

# Commands
## Run Optimization

To compute optimized stint strategies for selected races and drivers:

```bash
python model.py
```

- Outputs are saved to:
  - `race_results_summary.csv` (stint breakdown + optimized total time)

---

## Compare with Real Race Data

To compare optimized strategy vs actual strategy using real F1 lap data:

```bash
python comparison.py
```

- Outputs:
  - `comparison_summary.csv` with real time, optimized time, and % gain

---

## Visualize Results

To generate a bar chart of real vs. optimized race times with % gain:

```bash
python result_chart.py
```

- Shows real vs optimized bars
- Displays % gain in a side summary box

---

#  File Structure

```
.
├── cache/                        # Required FastF1 cache
├── model.py                     # Optimization model
├── f1_data.py                   # Data extraction and preprocessing
├── comparison.py                # Real vs optimized comparison
├── result_chart.py              # Bar chart visualizer
├── race_results_summary.csv     # Output: optimized strategies
├── comparison_summary.csv       # Output: real vs optimized comparison
└── requirements.txt             # Python dependencies
```

---

 # Summary

This project formulates F1 race strategy as a convex optimization problem with:
- Objective: Minimize total race time
- Constraints: tire usage, stint length, pit stop cost
- Penalties: fuel load, degradation, pit stop time

It demonstrates how mathematical modeling can outperform real-world strategies under idealized assumptions.

---

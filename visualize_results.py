import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the results CSV
df = pd.read_csv('race_results_summary.csv')

#  new column for combined Race + Year
df['Race (Year)'] = df['Race'] + ' ' + df['Year'].astype(str)

# Get unique races and drivers
races = df['Race (Year)'].unique()
drivers = df['Driver'].unique()

# bar chart parameters
bar_width = 0.25
x = np.arange(len(races))

#  mapping from race to index
race_to_idx = {race: idx for idx, race in enumerate(races)}

# bar data per driver
fig, ax = plt.subplots(figsize=(14, 6))

for i, driver in enumerate(drivers):
    driver_data = df[df['Driver'] == driver]
    heights = []
    positions = []

    for race in races:
        row = driver_data[driver_data['Race (Year)'] == race]
        if not row.empty:
            heights.append(float(row['Total Time (s)'].values[0]))
        else:
            heights.append(0)  
        positions.append(race_to_idx[race] + i * bar_width)

    ax.bar(positions, heights, width=bar_width, label=driver)

# Set up axis and labels
ax.set_xlabel('Race')
ax.set_ylabel('Total Optimized Time (s)')
ax.set_title('Optimized Race Time by Driver and Race')
ax.set_xticks([r + bar_width for r in range(len(races))])
ax.set_xticklabels(races, rotation=45, ha='right')
ax.legend()
plt.tight_layout()
#plt.savefig('race_strategy_comparison_matplotlib.png', dpi=300)
plt.show()


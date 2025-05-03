import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('comparison_summary.csv')
df.columns = df.columns.str.strip()

# Combining race and driver for labeling
df['Label'] = df['Race'] + ' - ' + df['Driver']

labels = df['Label']
real_times = df['Real Time (s)']
optimized_times = df['Optimized Time (s)']
percent_gains = df['% Gain']

x = np.arange(len(labels))
bar_width = 0.4

fig, ax = plt.subplots(figsize=(15, 6))

# Plot bars
real_bars = ax.bar(x - bar_width/2, real_times, bar_width, label='Real Time', color='skyblue')
opt_bars = ax.bar(x + bar_width/2, optimized_times, bar_width, label='Optimized Time', color='lightgreen')
for bars in [real_bars, opt_bars]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

#  % Gain text block 
gain_text = "➤ % Gain per Driver:\n\n"
for label, gain in zip(labels, percent_gains):
    if pd.notna(gain):
        gain_text += f"{label}: {gain:.2f}%\n"
props = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9)
ax.text(1.02, 0.5, gain_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='center', bbox=props)


ax.set_xlabel('Race & Driver')
ax.set_ylabel('Race Time (seconds)')
ax.set_title('Real vs Optimized Race Time per Driver')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend()
plt.subplots_adjust(bottom=0.25, right=0.75)
plt.show()

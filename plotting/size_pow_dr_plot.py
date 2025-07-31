import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('./variable_importance/size_power_dr.csv')

# Compute the true dropout rate (what BRATD actually used)
df['dropout_rate'] = 1.0 - df['dropout_rate']

# Sort by the true dropout for a clean plot
df = df.sort_values('dropout_rate')

# Now plot against the corrected axis
plt.style.use('matplotlibrc')
x = df['dropout_rate']
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, df['type_I_mean'],  marker='o', label='Type I Error')
ax.plot(x, df['type_II_mean'], marker='o', label='Type II Error')
ax.fill_between(x,
                df['type_I_mean']  - 2*df['type_I_std'],
                df['type_I_mean']  + 2*df['type_I_std'],
                alpha=0.1)
ax.fill_between(x,
                df['type_II_mean'] - 2*df['type_II_std'],
                df['type_II_mean'] + 2*df['type_II_std'],
                alpha=0.1)

ax.set_xlabel('Dropout Rate')
ax.set_ylabel('Error Rate')
ax.set_title('Type I & Type II Error vs Dropout Rate (±2 ste)')
ax.legend()
plt.tight_layout()
plt.show()


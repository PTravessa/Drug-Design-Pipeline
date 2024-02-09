import pandas as pd
import matplotlib.pyplot as plt

# Read SASA data
df_sasa = pd.read_csv('SASA.xvg', delimiter='\s+', skiprows=24, names=['Time', 'SASA'])

# Convert time to nanoseconds
df_sasa['Time'] /= 1000  # Convert ps to ns

# Save all SASA data to CSV with commas as separators
df_sasa.to_csv('sasa_data_ns.csv', index=False, sep=',')

# Basic statistics
mean_sasa = df_sasa['SASA'].mean()
min_sasa = df_sasa['SASA'].min()
max_sasa = df_sasa['SASA'].max()

# Plot SASA over time
plt.figure(figsize=(10, 6))
plt.plot(df_sasa['Time'], df_sasa['SASA'], label='SASA')
plt.xlabel('Time (ns)')
plt.ylabel('SASA (nm^2)')
plt.title('Solvent Accessible Surface Area (SASA) over Time')
plt.legend()
plt.savefig('sasa_plot_ns.png')

# Save results to CSV with commas as separators
result_dict = {
    'Mean SASA': [mean_sasa],
    'Min SASA': [min_sasa],
    'Max SASA': [max_sasa]
}
df_results = pd.DataFrame(result_dict)
df_results.to_csv('sasa_results_ns.csv', index=False, sep=',')

# Display summary
print("SASA Analysis Results:")
print(f"Mean SASA: {mean_sasa} nm^2")
print(f"Min SASA: {min_sasa} nm^2")
print(f"Max SASA: {max_sasa} nm^2")

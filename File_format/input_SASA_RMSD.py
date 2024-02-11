import pandas as pd

# Define the file path for SASA analysis
sasa_file = 'LPLPLPL_SASA.xvg'

# Read the SASA analysis file into a DataFrame, skipping 18 rows and setting the time unit to ns
df_sasa = pd.read_csv(sasa_file, skiprows=24, delimiter='\s+', names=['Time (ns)', 'SASA (nm^2)'])

# Convert time from ps to ns
df_sasa['Time (ns)'] /= 1000

# Extract peptide designation from the file name
peptide_designation = sasa_file.split('_')[0]

# Add a column for peptide designation
df_sasa['Peptide Designation'] = peptide_designation

# Save the DataFrame to a CSV file
output_file_sasa = f'{peptide_designation}_SASA.csv'
df_sasa.to_csv(output_file_sasa, index=False, sep=',', float_format='%.7f', columns=['Peptide Designation', 'SASA (nm^2)'])

# Define the file path for RMSD analysis
rmsd_file = 'LPLPLPL_RMSD.xvg'

# Read the RMSD analysis file into a DataFrame, skipping 18 rows and setting the time unit to ns
df_rmsd = pd.read_csv(rmsd_file, skiprows=18, delimiter='\s+', names=['Time (ns)', 'RMSD (nm)'])

# Convert time from ps to ns
df_rmsd['Time (ns)'] /= 1000

# Extract peptide designation from the file name
peptide_designation = rmsd_file.split('_')[0]

# Add a column for peptide designation
df_rmsd['Peptide Designation'] = peptide_designation

# Save the DataFrame to a CSV file
output_file_rmsd = f'{peptide_designation}_RMSD.csv'
df_rmsd.to_csv(output_file_rmsd, index=False, sep=',', float_format='%.7f', columns=['Peptide Designation', 'RMSD (nm)'])

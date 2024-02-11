import pandas as pd

# usage for SASA descriptor
sasa_file = 'LPLPLPL_SASA.xvg'
sasa_skip_rows = 24
sasa_column_names = ['Time (ns)', 'SASA']
sasa_time_unit = 1000  # Convert ps to ns
sasa_output_column_name = 'SASA'


# usage for RMSD descriptor
rmsd_file = 'LPLPLPL_RMSD.xvg'
rmsd_skip_rows = 18
rmsd_column_names = ['Time (ns)', 'RMSD']
rmsd_time_unit = 1000  # Convert ps to ns
rmsd_output_column_name = 'RMSD'

def read_process_save_data(input_file, skip_rows, column_names, time_unit, output_column_name):
    try:
        # Read the analysis file into a DataFrame
        df = pd.read_csv(input_file, skiprows=skip_rows, delimiter='\s+', names=column_names)

        # Convert time to the specified unit
        df['Time (ns)'] /= time_unit

        # Extract peptide designation from the file name
        peptide_designation = input_file.split('_')[0]

        # Add a column for peptide designation
        df['Peptide Designation'] = peptide_designation

        # Save the DataFrame to a CSV file
        output_file = f'{peptide_designation}_{output_column_name}.csv'
        df.to_csv(output_file, index=False, sep=',', float_format='%.7f',
                  columns=['Peptide Designation','Time (ns)', output_column_name])
        print(f"Data saved to '{output_file}'")

        return df  # Return the processed DataFrame
    except Exception as e:
        print(f"Error processing file '{input_file}': {e}")
        return None  # Return None if an error occurs

def combine_descriptors(descriptor_data):
    try:
        # Check if all descriptor dataframes are available
        if all(df is not None for df in descriptor_data):
            # Concatenate all descriptor dataframes along the columns axis
            df_combined = pd.concat(descriptor_data, axis=1)

            # Save the combined dataframe to a CSV file
            combined_output_file = 'combined_descriptors.csv'
            df_combined.to_csv(combined_output_file, index=False, float_format='%.7f')
            print(f"Combined data saved to '{combined_output_file}'")
        else:
            print("Error: Unable to concatenate dataframes due to missing data.")
    except Exception as e:
        print(f"Error combining descriptors: {e}")


df_rmsd = read_process_save_data(rmsd_file, rmsd_skip_rows, rmsd_column_names, rmsd_time_unit, rmsd_output_column_name)
df_sasa = read_process_save_data(sasa_file, sasa_skip_rows, sasa_column_names, sasa_time_unit, sasa_output_column_name)
# Combine descriptors
combine_descriptors([df_sasa[['Peptide Designation', 'Time (ns)', 'SASA']], df_rmsd[['RMSD']]])
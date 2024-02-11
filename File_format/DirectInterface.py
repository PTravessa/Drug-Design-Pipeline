import pandas as pd
import os

def read_process_save_data(input_file, skip_rows, column_names, time_unit, output_column_name):
    try:
        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"File '{input_file}' not found.")

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

        return output_file, df  # Return the output file name and the processed DataFrame
    except Exception as e:
        print(f"Error processing file '{input_file}': {e}")
        return None, None  # Return None if an error occurs


def print_columns_info(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Columns in {file_path}:")
        for i, column in enumerate(df.columns, start=1):
            print(f"{i}: {column}")
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")


def combine_descriptors(sasa_csv, rmsd_csv, selected_sasa_columns, selected_rmsd_columns):
    try:
        df_sasa = pd.read_csv(sasa_csv)
        df_rmsd = pd.read_csv(rmsd_csv)

        # Select columns from SASA and RMSD based on user input
        selected_sasa_columns = [df_sasa.columns[int(i)-1] for i in selected_sasa_columns.split(',')]
        selected_rmsd_columns = [df_rmsd.columns[int(i)-1] for i in selected_rmsd_columns.split(',')]

        # Combine selected columns from SASA and RMSD dataframes
        df_combined = pd.concat([df_sasa[selected_sasa_columns], df_rmsd[selected_rmsd_columns]], axis=1)

        # Save the combined dataframe to a CSV file
        combined_output_file = 'combined_descriptors.csv'
        df_combined.to_csv(combined_output_file, index=False, float_format='%.7f')
        print(f"Combined data saved to '{combined_output_file}'")
    except Exception as e:
        print(f"Error combining descriptors: {e}")


def main():
    # Define parameters for SASA analysis
    sasa_file = 'LPLPLPL_SASA.xvg'
    sasa_skip_rows = 24
    sasa_column_names = ['Time (ns)', 'SASA']
    sasa_time_unit = 1000  # Convert ps to ns
    sasa_output_column_name = 'SASA'

    # Define parameters for RMSD analysis
    rmsd_file = 'LPLPLPL_RMSD.xvg'
    rmsd_skip_rows = 18
    rmsd_column_names = ['Time (ns)', 'RMSD']
    rmsd_time_unit = 1000  # Convert ps to ns
    rmsd_output_column_name = 'RMSD'

    # Read, process, and save SASA data
    sasa_output_csv, df_sasa = read_process_save_data(sasa_file, sasa_skip_rows, sasa_column_names, sasa_time_unit, sasa_output_column_name)

    # Read, process, and save RMSD data
    rmsd_output_csv, df_rmsd = read_process_save_data(rmsd_file, rmsd_skip_rows, rmsd_column_names, rmsd_time_unit, rmsd_output_column_name)

    if df_sasa is not None and df_rmsd is not None:
        print_columns_info(sasa_output_csv)
        selected_sasa_columns = input("Enter indices of columns from SASA CSV (comma-separated): ")
        print_columns_info(rmsd_output_csv)
        selected_rmsd_columns = input("Enter indices of columns from RMSD CSV (comma-separated): ")

        combine_descriptors(sasa_output_csv, rmsd_output_csv, selected_sasa_columns, selected_rmsd_columns)
    else:
        print("Error: Unable to concatenate dataframes due to missing data.")


if __name__ == "__main__":
    main()

import pandas as pd

# Load the Excel file
file_path = "Data Cat.xlsx"
xls = pd.ExcelFile(file_path)

# Display sheet names to understand the structure
xls.sheet_names

# Load the data sheet
df = pd.read_excel(xls, sheet_name="Data")

# Display the first few rows to inspect the structure
df.head()
# Identify the actual header row (usually the first meaningful row in financial reports)
df_cleaned = pd.read_excel(xls, sheet_name="Data", skiprows=6)  # Skipping metadata rows

# Drop completely empty columns
df_cleaned = df_cleaned.dropna(axis=1, how='all')

# Display first few rows after cleanup
df_cleaned.head()
# Identify the first valid row that contains column headers
df_cleaned.dropna(how='all', inplace=True)  # Remove fully empty rows
df_cleaned.reset_index(drop=True, inplace=True)  # Reset index

# Set the first row as the header
df_cleaned.columns = df_cleaned.iloc[0]
df_cleaned = df_cleaned[1:].reset_index(drop=True)

# Drop any remaining unnamed columns with NaN headers
df_cleaned = df_cleaned.loc[:, ~df_cleaned.columns.isna()]

# Display cleaned column names
df_cleaned.head()
print(df_cleaned.columns)

# Export to CSV
output_file = "cleaned_data.csv"
df_cleaned.to_csv(output_file, index=False)
print(f"\nData has been exported to {output_file}")

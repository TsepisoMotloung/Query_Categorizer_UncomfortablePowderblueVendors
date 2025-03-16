
import pandas as pd

# Load the Excel file
file_path = "Data Cat.xlsx"
df = pd.read_excel(file_path, sheet_name="Data", skiprows=6)

# Clean the data
df = df.dropna(axis=1, how='all')
df.dropna(how='all', inplace=True)
df.reset_index(drop=True, inplace=True)

# Set headers and clean data
df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)
df = df.loc[:, ~df.columns.isna()]

# Remove 'budget' columns and clean column names
columns_to_keep = [col for col in df.columns if 'budget' not in str(col).lower()]
df = df[columns_to_keep]

# Define the correct order of months (Jan 23 to Dec 24)
month_order = [
    'Jan 23', 'Feb 23', 'Mar 23', 'Apr 23', 'May 23', 'Jun 23', 
    'Jul 23', 'Aug 23', 'Sep 23', 'Oct 23', 'Nov 23', 'Dec 23',
    'Jan 24', 'Feb 24', 'Mar 24', 'Apr 24', 'May 24', 'Jun 24', 
    'Jul 24', 'Aug 24', 'Sep 24', 'Oct 24', 'Nov 24', 'Dec 24'
]

# Clean column names and standardize format
def clean_column_name(col):
    col = str(col).strip()
    if any(month in col.lower() for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
        # Extract month and year
        for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
            if month.lower() in col.lower():
                year = '23' if '2023' in col or '23' in col else '24'
                return f"{month} {year}"
    return col

# Clean and rename columns
df.columns = [clean_column_name(col) for col in df.columns]

# Identify non-month columns (like PaypointName, etc.)
non_month_columns = [col for col in df.columns if col not in month_order]

# Reorder columns: non-month columns first, then months in chronological order
available_months = [m for m in month_order if m in df.columns]
final_column_order = non_month_columns + available_months

# Reorder the columns
df = df[final_column_order]

# Export to CSV
output_file = "cleaned_data.csv"
df.to_csv(output_file, index=False)
print(f"Data has been cleaned and exported to {output_file}")
print("\nColumn names in the cleaned data:")
print(df.columns.tolist())

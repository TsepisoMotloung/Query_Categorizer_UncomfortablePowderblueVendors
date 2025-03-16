
import pandas as pd

# Load the Excel file
file_path = "Data Cat.xlsx"
df = pd.read_excel(file_path, sheet_name="Data", skiprows=6)

# Clean the data
df = df.dropna(axis=1, how='all')
df.dropna(how='all', inplace=True)
df.reset_index(drop=True, inplace=True)

# Identify the first row with actual data (not headers)
first_data_row = df.index[df.iloc[:, 0].notna()][0]

# Set proper column names from the row before the first data row
df.columns = df.iloc[first_data_row - 1]
df = df.iloc[first_data_row:].reset_index(drop=True)

# Remove budget columns
columns_to_keep = [col for col in df.columns if 'budget' not in str(col).lower()]
df = df[columns_to_keep]

# Define the correct order of months
month_order = [
    'Jan 23', 'Feb 23', 'Mar 23', 'Apr 23', 'May 23', 'Jun 23', 
    'Jul 23', 'Aug 23', 'Sep 23', 'Oct 23', 'Nov 23', 'Dec 23',
    'Jan 24', 'Feb 24', 'Mar 24', 'Apr 24', 'May 24', 'Jun 24', 
    'Jul 24', 'Aug 24', 'Sep 24', 'Oct 24', 'Nov 24', 'Dec 24'
]

# Clean column names
def clean_column_name(col):
    col = str(col).strip()
    for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
        if month.lower() in col.lower():
            if '2023' in col or '23' in col:
                return f"{month} 23"
            elif '2024' in col or '24' in col:
                return f"{month} 24"
    return col

# Clean column names
df.columns = [clean_column_name(col) for col in df.columns]

# Separate month and non-month columns
month_columns = [col for col in df.columns if any(month in col for month in month_order)]
non_month_columns = [col for col in df.columns if col not in month_columns]

# Sort month columns according to month_order
sorted_month_columns = [m for m in month_order if m in month_columns]

# Combine columns in correct order
final_column_order = non_month_columns + sorted_month_columns
df = df[final_column_order]

# Export to CSV
output_file = "cleaned_data.csv"
df.to_csv(output_file, index=False)
print(f"Data has been cleaned and exported to {output_file}")
print("\nColumns in the cleaned data:")
print(df.columns.tolist())

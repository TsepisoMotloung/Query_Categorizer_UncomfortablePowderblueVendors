
import pandas as pd

# Load the Excel file
file_path = "Category New Business Queries.xlsx"
xls = pd.ExcelFile(file_path)

# Display sheet names to understand the structure
print(xls.sheet_names)

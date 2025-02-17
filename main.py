
import pandas as pd

try:
    # Load the Excel file
    file_path = "Category New Business Queries.xlsx"
    xls = pd.ExcelFile(file_path)
    
    # Display sheet names
    print("Available sheets in the Excel file:")
    for i, sheet in enumerate(xls.sheet_names, 1):
        print(f"{i}. {sheet}")

except FileNotFoundError:
    print(f"Error: Could not find the Excel file '{file_path}'")
except Exception as e:
    print(f"Error: {str(e)}")

import pandas as pd

# Load the Excel file
file_path = "Product Categorizer.xlsx"
xls = pd.ExcelFile(file_path)

# Display sheet names to understand the structure
xls.sheet_names

# Load the sheet into a DataFrame
df = pd.read_excel(xls, sheet_name="Sheet1")

# Display the first few rows to understand the structure
df.head()

import re

# Function to extract product names from the Ref column
def extract_product_name(ref):
    match = re.search(r"(?i)(?:Billing.*?\d{4}-\d{2}-\d{2})?\s*([\w\s-]+)", ref)
    return match.group(1).strip() if match else ref

# Apply the extraction function
df["Extracted Product"] = df["Ref"].astype(str).apply(extract_product_name)

# Display the updated dataframe
df.head()

# Improved extraction function to better capture product names
def refine_product_extraction(ref):
    # Remove numeric codes and billing dates
    clean_ref = re.sub(r"\b\d{4,5}\/?\d*\b|\bBilling.*?\d{4}-\d{2}-\d{2}\b", "", ref, flags=re.IGNORECASE)
    # Remove common words like "Billing Alignment"
    clean_ref = re.sub(r"\b(Billing Alignment|Billing)\b", "", clean_ref, flags=re.IGNORECASE).strip()
    return clean_ref

# Apply refined extraction
df["Extracted Product"] = df["Ref"].astype(str).apply(refine_product_extraction)

# Display updated dataframe
df.head()


# Export to CSV files
df.to_csv('full_data.csv', index=False)
df[['Ref', 'Extracted Product']].to_csv('extracted_products.csv', index=False)

print("\nData exported successfully to:")
print("1. full_data.csv - Contains all columns")
print("2. extracted_products.csv - Contains only Ref and Extracted Product columns")


print(df)
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


# Export to Excel
output_path = "Processed_Products.xlsx"


# Export with specific columns
df[['Ref', 'Extracted Product']].to_excel(output_path, index=False)

# Export to a specific sheet name
with pd.ExcelWriter(output_path) as writer:
    df.to_excel(writer, sheet_name='Processed Data', index=False)

# Export with multiple sheets
with pd.ExcelWriter(output_path) as writer:
    df.to_excel(writer, sheet_name='Full Data', index=False)
    df[['Ref', 'Extracted Product']].to_excel(writer, sheet_name='Extracted Products', index=False)


print(df)
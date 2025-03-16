
import pandas as pd
import os

# Load the Excel file
file_path = "Data Cat.xlsx"
df = pd.read_excel(file_path, sheet_name="Data", skiprows=6)

# Clean the data
df = df.dropna(axis=1, how='all')
df.dropna(how='all', inplace=True)
df.reset_index(drop=True, inplace=True)
df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)
df = df.loc[:, ~df.columns.isna()]

def categorize_paypoint(name):
    name_lower = str(name).lower()
    if "cash" in name_lower:
        return "Cash"
    elif "debit order" in name_lower:
        import re
        number = re.search(r'debit order\s*(\d+)', name_lower)
        if number:
            return f"Debit Order {number.group(1)}"
        return "Debit Order"
    elif any(term in name_lower for term in ["government stop order", "ministry", "judiciary"]):
        return "Government Stop Order"
    elif "payment deduction" in name_lower:
        import re
        number = re.search(r'payment deduction\s*(\d+)', name_lower)
        if number:
            return f"Payment Deduction {number.group(1)}"
        return "Payment Deduction"
    else:
        return "Uncategorized"

# Apply categorization
df['Category'] = df['PaypointName'].apply(categorize_paypoint)

# Save to CSV
output_file = "categorized_paypoints.csv"
df.to_csv(output_file, index=False)
print(f"Data has been saved to {output_file}")

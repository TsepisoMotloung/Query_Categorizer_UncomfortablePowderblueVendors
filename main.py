# Load the first sheet into a DataFrame and display the first few rows
df = pd.read_excel(xls, sheet_name="sheet1")
df.head()

# Define categorization function
def categorize_paypoint(name):
    name_lower = str(name).lower()
    if "cash" in name_lower:
        return "Cash"
    elif "debit order" in name_lower or "debit order 01" in name_lower:
        return "Debit Order O1"
    elif "government stop order" in name_lower:
        return "Government Stop Order"
    elif "payment deduction" in name_lower:
        return "Payment Deduction"
    else:
        return "Uncategorized"

# Apply categorization
df["Category"] = df["PaypointName(New)"].apply(categorize_paypoint)

# Show results
df[["PaypointName(New)", "Category"]].head(10)

# Define CSV file path
csv_file_path = "/mnt/data/categorized_paypoints.csv"

# Save the categorized data to CSV
df[["PaypointName(New)", "Category"]].to_csv(csv_file_path, index=False)

# Provide download link
csv_file_path

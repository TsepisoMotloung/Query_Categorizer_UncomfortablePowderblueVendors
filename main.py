
import pandas as pd
import os

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
        # Extract any numbers that follow "payment deduction"
        import re
        number = re.search(r'payment deduction\s*(\d+)', name_lower)
        if number:
            return f"Payment Deduction {number.group(1)}"
        return "Payment Deduction"
    else:
        return "Uncategorized"

def main():
    # Read the Excel file
    excel_file = "0124430 Do Not Use Paypoint (1).xlsx"
    if not os.path.exists(excel_file):
        print(f"Error: {excel_file} not found")
        return

    # Load the first sheet into a DataFrame
    df = pd.read_excel(excel_file, sheet_name="sheet1")
    print("First few rows of the original data:")
    print(df.head())

    # Apply categorization
    df["Category"] = df["PaypointName(New)"].apply(categorize_paypoint)

    # Show results
    print("\nSample of categorized data:")
    print(df[["PaypointName(New)", "Category"]].head(10))

    # Define CSV file path
    csv_file_path = "categorized_paypoints.csv"

    # Save the categorized data to CSV
    df[["PaypointName(New)", "Category"]].to_csv(csv_file_path, index=False)
    print(f"\nData has been saved to {csv_file_path}")

if __name__ == "__main__":
    main()

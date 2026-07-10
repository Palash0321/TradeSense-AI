import pandas as pd

# Read current files
india = pd.read_csv("data/india_stocks.csv")
usa = pd.read_csv("data/us_stocks.csv")

# Clean spaces
india["symbol"] = india["symbol"].astype(str).str.strip().str.upper()
india["name"] = india["name"].astype(str).str.strip()

usa["symbol"] = usa["symbol"].astype(str).str.strip().str.upper()
usa["name"] = usa["name"].astype(str).str.strip()

# Remove duplicate symbols
india = india.drop_duplicates(subset="symbol")
usa = usa.drop_duplicates(subset="symbol")

# Merge
master = pd.concat([india, usa], ignore_index=True)

# Remove duplicates again
master = master.drop_duplicates(subset="symbol")

# Sort
master = master.sort_values(["market", "symbol"])

# Save
master.to_csv("data/master_stocks.csv", index=False)

print("=" * 50)
print(f"Indian Stocks : {len(india)}")
print(f"US Stocks     : {len(usa)}")
print(f"Master Stocks : {len(master)}")
print("=" * 50)
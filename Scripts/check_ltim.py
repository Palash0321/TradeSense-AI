import pandas as pd

df = pd.read_csv("data/india_stocks.csv")

print(df[df["name"].str.contains("Mindtree", case=False, na=False)])
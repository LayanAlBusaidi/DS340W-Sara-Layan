import pandas as pd

df = pd.read_csv(
    "data/accomodation/raw/new_york/listings_2026-09-14.csv.gz",
    compression="gzip"
)

print(df.head())
print(df.shape)
print(df.columns.tolist())
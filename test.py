import pandas as pd
df = pd.read_csv("data/amazon.csv")

print(df.shape)

print(df.columns.tolist())
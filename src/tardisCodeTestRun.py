import pandas as pd

# Reading directly from a .csv.gz file
data = pd.read_csv(r'C:\workspace\dqn\mywork\datasets\binance_book_snapshot_25_2023-04-14_BTCUSDT.csv.gz', compression='gzip')

print(data)
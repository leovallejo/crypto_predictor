import pandas as pd
from tabulate import tabulate
import os
import time
from config import *

def display_predictions():
    try:
        df = pd.read_csv(PREDICTION_HISTORY_FILE)
        if df.empty:
            print("No predictions yet")
            return
            
        latest = df.sort_values('timestamp').groupby(['symbol', 'timeframe']).last().reset_index()
        
        display_df = latest[['timestamp', 'symbol', 'timeframe', 'current']].copy()
        display_df['prediction'] = latest['predictions'].apply(lambda x: eval(x)[-1])
        display_df['change%'] = 100*(display_df['prediction']/display_df['current'] - 1)
        
        print(tabulate(
            display_df.round(2),
            headers=['Time', 'Symbol', 'TF', 'Current', 'Prediction', 'Change%'],
            tablefmt='simple_grid',
            showindex=False
        ))
        
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    print("🔄 Live Crypto Monitor (CPU Mode)")
    while True:
        display_predictions()
        time.sleep(REFRESH_INTERVAL)
        os.system('clear')

if __name__ == "__main__":
    main()

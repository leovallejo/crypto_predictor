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
        
        # Format table
        display_df = latest[['timestamp', 'symbol', 'timeframe', 'current']].copy()
        display_df['pred_5'] = latest['predictions'].apply(lambda x: eval(x)[4])
        display_df['pred_10'] = latest['predictions'].apply(lambda x: eval(x)[9])
        display_df['pred_final'] = latest['predictions'].apply(lambda x: eval(x)[-1])
        
        print(tabulate(
            display_df.round(2),
            headers=['Time', 'Symbol', 'TF', 'Current', '+5', '+10', 'Final'],
            tablefmt='fancy_grid',
            showindex=False
        ))
        
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    print("🔄 Live Swing Trade Monitor")
    while True:
        display_predictions()
        time.sleep(REFRESH_INTERVAL)
        os.system('cls' if os.name == 'nt' else 'clear')

if __name__ == "__main__":
    main()

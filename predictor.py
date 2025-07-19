import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from tensorflow.keras.models import load_model
from config import *

class SwingPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def load_models(self):
        for timeframe in TIMEFRAMES:
            for token in SUPPORTED_TOKENS:
                tag = f"{token}_{timeframe}"
                try:
                    self.models[tag] = load_model(os.path.join(MODEL_DIR, f"{tag}_model.h5"))
                    self.scalers[tag] = {
                        'feature': joblib.load(os.path.join(MODEL_DIR, f"{tag}_feat_scaler.pkl")),
                        'close': joblib.load(os.path.join(MODEL_DIR, f"{tag}_close_scaler.pkl"))
                    }
                except Exception as e:
                    print(f"Error loading {tag}: {str(e)}")

    def predict(self, token, timeframe):
        tag = f"{token}_{timeframe}"
        df = fetch_binance_data(f"{token}{FIAT_CURRENCY}", timeframe, SEQUENCE_LENGTH+10)
        df = add_technical_indicators(df)
        
        features = self.scalers[tag]['feature'].transform(df.tail(SEQUENCE_LENGTH))
        pred = self.models[tag].predict(features[np.newaxis, ...])[0]
        pred_prices = self.scalers[tag]['close'].inverse_transform(pred.reshape(-1, 1)).flatten()
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': token,
            'timeframe': timeframe,
            'current': df['close'].iloc[-1],
            'predictions': pred_prices.tolist()
        }

if __name__ == "__main__":
    predictor = SwingPredictor()
    predictor.load_models()
    while True:
        predictions = []
        for timeframe in TIMEFRAMES:
            for token in SUPPORTED_TOKENS:
                predictions.append(predictor.predict(token, timeframe))
        pd.DataFrame(predictions).to_csv(PREDICTION_HISTORY_FILE, mode='a', header=not os.path.exists(PREDICTION_HISTORY_FILE))
        time.sleep(PREDICTION_INTERVAL)

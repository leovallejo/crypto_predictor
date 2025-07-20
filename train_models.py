import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress warnings

import numpy as np
import pandas as pd
import requests
import joblib
import logging
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, LSTM, Dense, Dropout)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
from tensorflow.keras.losses import Huber
from sklearn.preprocessing import RobustScaler
import keras_tuner as kt
from config import *
import time
import random

# Configure logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Suppress TensorFlow and Keras-Tuner warnings
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

def send_telegram_message(message):
    """Simplified Telegram notification"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = TELEGRAM_API_URL.format(token=TELEGRAM_BOT_TOKEN)
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
    except Exception as e:
        logger.error(f"Telegram error: {e}")

def fetch_binance_data(symbol, interval, limit=DATA_FETCH_LIMIT):
    """Fetch data with retries and simplified headers"""
    for attempt in range(3):
        try:
            response = requests.get(
                BINANCE_API_URL,
                params={'symbol': symbol, 'interval': interval, 'limit': limit},
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=15
            )
            response.raise_for_status()
            
            df = pd.DataFrame(
                response.json(),
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )[[0, 1, 2, 3, 4, 5]]
            df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df.astype(float)
            
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

# [Rest of your functions (calculate_rsi, add_technical_indicators, etc.) remain similar but with reduced complexity]

def build_model(hp):
    """CPU-optimized model architecture"""
    model = Sequential([
        Conv1D(
            filters=hp.Int('conv_filters', 32, 96, step=32),
            kernel_size=hp.Int('kernel_size', 3, 5, step=1),
            activation='relu',
            input_shape=(SEQUENCE_LENGTH, len(TECHNICAL_INDICATORS) + 5)
        ),
        MaxPooling1D(2),
        LSTM(hp.Int('lstm_units', 32, 96, step=32)),
        Dropout(hp.Float('dropout', 0.1, 0.3)),
        Dense(FORECAST_HORIZON)
    ])
    
    model.compile(
        optimizer=Adam(hp.Float('lr', 1e-4, 1e-3)),
        loss=Huber(),
        metrics=['mae']
    )
    return model

def train_model(token, timeframe):
    """Simplified training function"""
    symbol = f"{token}{FIAT_CURRENCY}"
    tag = f"{token}_{timeframe}"
    model_path = os.path.join(MODEL_DIR, f"{tag}_model.h5")
    
    try:
        df = fetch_binance_data(symbol, timeframe)
        X_train, y_train, X_test, y_test, feat_scaler, close_scaler, _ = prepare_data(df)
        
        tuner = kt.Hyperband(
            build_model,
            objective='val_loss',
            max_epochs=MAX_TRAIN_EPOCHS,
            factor=3,
            directory=os.path.join(MODEL_DIR, 'tuning'),
            project_name=tag
        )
        
        tuner.search(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=BATCH_SIZE,
            callbacks=[
                EarlyStopping(patience=EARLY_STOPPING_PATIENCE),
                ReduceLROnPlateau(factor=0.5, patience=3)
            ]
        )
        
        best_model = tuner.get_best_models()[0]
        best_model.save(model_path)
        joblib.dump(feat_scaler, os.path.join(MODEL_DIR, f"{tag}_feat_scaler.pkl"))
        joblib.dump(close_scaler, os.path.join(MODEL_DIR, f"{tag}_close_scaler.pkl"))
        
        send_telegram_message(f"✅ {tag} trained")
        
    except Exception as e:
        send_telegram_message(f"❌ {tag} failed: {str(e)}")
        raise

def main():
    send_telegram_message("🏁 Starting training")
    for timeframe in TIMEFRAMES:
        for token in SUPPORTED_TOKENS:
            train_model(token, timeframe)
    send_telegram_message("🎉 Training complete")

if __name__ == "__main__":
    main()

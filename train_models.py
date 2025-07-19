import os
import numpy as np
import pandas as pd
import requests
import joblib
import logging
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, LSTM, Dense, 
                                   Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, 
                                      ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Huber
from sklearn.preprocessing import RobustScaler
import keras_tuner as kt
from config import *

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Initialize logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)

def configure_gpu():
    """Configure GPU settings"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU configured: {gpus}")
            return True
        except RuntimeError as e:
            logger.error(f"GPU config error: {e}")
    return False

configure_gpu()

def send_telegram_message(message):
    """Send notifications via Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = TELEGRAM_API_URL.format(token=TELEGRAM_BOT_TOKEN)
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        logger.error(f"Telegram error: {e}")

def fetch_binance_data(symbol, interval, limit=DATA_FETCH_LIMIT):
    """Fetch OHLCV data from Binance API"""
    try:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        response = requests.get(BINANCE_API_URL, params=params, timeout=15)
        response.raise_for_status()
        
        df = pd.DataFrame(
            response.json(),
            columns=["timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_av", "trades", "tb_base_av",
                    "tb_quote_av", "ignore"]
        )
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df.astype(float)
    except Exception as e:
        logger.error(f"Error fetching {symbol} data: {e}")
        raise

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def add_technical_indicators(df):
    """Add technical indicators to dataframe"""
    # Price transformations
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['close_5ma'] = df['close'].rolling(5).mean()
    
    # Moving Averages
    for ma in TECHNICAL_INDICATORS['MA']:
        df[f'MA_{ma}'] = df['close'].rolling(ma).mean()
    
    # Exponential MAs
    for ema in TECHNICAL_INDICATORS['EMA']:
        df[f'EMA_{ema}'] = df['close'].ewm(span=ema, adjust=False).mean()
    
    # RSI
    df['RSI_14'] = calculate_rsi(df['close'])
    
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_MA20'] = df['close'].rolling(20).mean()
    df['BB_UPPER'] = df['BB_MA20'] + 2 * df['close'].rolling(20).std()
    df['BB_LOWER'] = df['BB_MA20'] - 2 * df['close'].rolling(20).std()
    
    return df.dropna()

def create_sequences(data, seq_length, horizon):
    """Create time-series sequences"""
    X, y = [], []
    for i in range(seq_length, len(data) - horizon):
        X.append(data[i-seq_length:i])
        y.append(data[i:i+horizon, 0])  # Predict close price only
    return np.array(X), np.array(y)

def prepare_data(df):
    """Prepare training data with scaling"""
    df = add_technical_indicators(df)
    feature_cols = [col for col in df.columns if col != 'timestamp']
    
    # Scaling
    feature_scaler = RobustScaler()
    close_scaler = RobustScaler()
    
    features = feature_scaler.fit_transform(df[feature_cols])
    close_prices = close_scaler.fit_transform(df[['close']])
    
    # Combine features and targets
    processed_data = np.concatenate([close_prices, features], axis=1)
    
    # Create sequences
    X, y = create_sequences(processed_data, SEQUENCE_LENGTH, FORECAST_HORIZON)
    
    # Train-test split
    split = int(len(X) * TRAIN_TEST_SPLIT)
    return X[:split], y[:split], X[split:], y[split:], feature_scaler, close_scaler, feature_cols

def build_model(hp):
    """Build CNN-LSTM model"""
    model = Sequential([
        Conv1D(
            hp.Int('conv_filters', 32, 256, step=32),
            hp.Int('kernel_size', 3, 9, step=2),
            activation='relu',
            input_shape=(SEQUENCE_LENGTH, len(FEATURES) + 1),
            kernel_regularizer=l2(hp.Float('l2_reg', 1e-5, 1e-3))
        ),
        BatchNormalization(),
        MaxPooling1D(2),
        LSTM(
            hp.Int('lstm_units', 64, 256, step=32),
            kernel_regularizer=l2(hp.Float('l2_reg', 1e-5, 1e-3))
        ),
        Dropout(hp.Float('dropout', 0.2, 0.5)),
        Dense(FORECAST_HORIZON)
    ])
    
    model.compile(
        optimizer=Adam(hp.Float('lr', 1e-5, 1e-3)),
        loss=Huber(),
        metrics=['mae']
    )
    return model

def train_model(token, timeframe):
    """Train model for specific token/timeframe"""
    symbol = f"{token}{FIAT_CURRENCY}"
    tag = f"{token}_{timeframe}"
    model_path = os.path.join(MODEL_DIR, f"{tag}_model.h5")
    
    if os.path.exists(model_path):
        logger.info(f"Skipping {tag} - already trained")
        return
    
    try:
        send_telegram_message(f"🚀 Training {tag}")
        df = fetch_binance_data(symbol, timeframe)
        X_train, y_train, X_test, y_test, feat_scaler, close_scaler, feat_cols = prepare_data(df)
        
        tuner = kt.Hyperband(
            build_model,
            objective='val_loss',
            max_epochs=MAX_TRAIN_EPOCHS,
            factor=3,
            directory=os.path.join(MODEL_DIR, 'tuning'),
            project_name=tag,
            overwrite=True
        )
        
        callbacks = [
            EarlyStopping(patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5),
            ModelCheckpoint(model_path, save_best_only=True)
        ]
        
        tuner.search(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save best model
        best_model = tuner.get_best_models()[0]
        best_model.save(model_path)
        joblib.dump(feat_scaler, os.path.join(MODEL_DIR, f"{tag}_feat_scaler.pkl"))
        joblib.dump(close_scaler, os.path.join(MODEL_DIR, f"{tag}_close_scaler.pkl"))
        
        send_telegram_message(f"✅ {tag} trained | Val loss: {tuner.oracle.get_best_trials()[0].score:.4f}")
        
    except Exception as e:
        send_telegram_message(f"❌ {tag} failed: {str(e)}")
        raise

def main():
    """Main training function"""
    send_telegram_message("🏁 Starting training pipeline")
    for timeframe in TIMEFRAMES:
        for token in SUPPORTED_TOKENS:
            train_model(token, timeframe)
    send_telegram_message("🎉 All models trained successfully")

if __name__ == "__main__":
    main()

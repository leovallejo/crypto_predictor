import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import requests
import joblib
import logging
from datetime import datetime, timedelta
import time
import random
from fake_useragent import UserAgent
import google.protobuf

# Import TensorFlow after environment variables
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, LSTM, Dense, Dropout)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
from tensorflow.keras.losses import Huber
from sklearn.preprocessing import RobustScaler
import keras_tuner as kt
from config import *

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Suppress warnings
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('keras-tuner').setLevel(logging.ERROR)

# Initialize fake user agent
ua = UserAgent()

def configure_environment():
    """Configure environment for CPU-only training"""
    tf.config.set_visible_devices([], 'GPU')
    print(f"Protobuf version: {google.protobuf.__version__}")
    print("Environment configured for CPU-only training")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras Tuner version: {kt.__version__}")
    print(f"Expected input shape: (None, {SEQUENCE_LENGTH}, {FEATURE_COUNT})")

configure_environment()

def send_telegram_message(message):
    """Send notifications via Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
        
    try:
        response = requests.post(
            TELEGRAM_API_URL.format(token=TELEGRAM_BOT_TOKEN),
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message},
            timeout=10
        )
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Telegram error: {str(e)}")

def fetch_binance_data(symbol, interval, limit=DATA_FETCH_LIMIT):
    """Fetch OHLCV data with multiple fallback strategies"""
    endpoints = [
        "https://api.binance.com/api/v3/klines",
        "https://api1.binance.com/api/v3/klines",
        "https://api2.binance.com/api/v3/klines",
        "https://api3.binance.com/api/v3/klines"
    ]
    
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    for attempt in range(5):
        try:
            endpoint = endpoints[attempt % len(endpoints)]
            headers = {'User-Agent': ua.random}
            
            response = requests.get(
                endpoint,
                params=params,
                headers=headers,
                timeout=15
            )
            
            if response.status_code == 451:
                return fetch_alternative_data(symbol, interval, limit)
                
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
            if attempt == 4:
                logger.error(f"Failed after 5 attempts for {symbol}/{interval}")
                raise Exception(f"All data sources failed for {symbol}/{interval}")
            time.sleep((attempt + 1) * 5)

def fetch_alternative_data(symbol, interval, limit):
    """Fallback data source when Binance API fails"""
    logger.warning(f"Using alternative data source for {symbol}/{interval}")
    
    try:
        from pycoingecko import CoinGeckoAPI
        cg = CoinGeckoAPI()
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=min(limit, 365))
        
        coin_id = 'bitcoin' if 'BTC' in symbol else 'ethereum'
        data = cg.get_coin_market_chart_range_by_id(
            id=coin_id,
            vs_currency='usd',
            from_timestamp=int(start_time.timestamp()),
            to_timestamp=int(end_time.timestamp())
        )
        
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['open'] = df['close']
        df['high'] = df['close']
        df['low'] = df['close']
        df['volume'] = 0
        
        return df.astype(float)
        
    except Exception as e:
        logger.error(f"Alternative data source failed: {str(e)}")
        raise Exception("All data sources failed")

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_macd(close_prices, fast=12, slow=26, signal=9):
    """Calculate MACD and Signal line"""
    ema_fast = close_prices.ewm(span=fast, adjust=False).mean()
    ema_slow = close_prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def add_technical_indicators(df):
    """Add technical indicators matching FEATURE_COUNT"""
    if len(df) < 30:
        raise ValueError("Insufficient data points for indicators")
    
    # Base columns (open, high, low, close, volume)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # Moving Averages
    for ma in TECHNICAL_INDICATORS['MA']:
        df[f'MA_{ma}'] = df['close'].rolling(ma).mean()
    
    # RSI
    if 'RSI' in TECHNICAL_INDICATORS:
        df['RSI'] = calculate_rsi(df['close'])
    
    # MACD
    if 'MACD' in TECHNICAL_INDICATORS:
        macd, signal = calculate_macd(df['close'])
        df['MACD'] = macd
        df['MACD_Signal'] = signal
    
    # Verify we have the expected number of features
    current_features = len(df.columns)
    expected_features = FEATURE_COUNT + 1  # +1 for timestamp
    
    if current_features != expected_features:
        # If we're missing features, add dummy columns to match expected count
        missing = expected_features - current_features
        for i in range(missing):
            df[f'dummy_{i}'] = 0.0
    
    return df.dropna()

def create_sequences(data, seq_length, horizon):
    """Create time-series sequences with shape verification"""
    X, y = [], []
    for i in range(seq_length, len(data) - horizon):
        X.append(data[i-seq_length:i])
        y.append(data[i:i+horizon, 0])  # Predict close price
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    # Verify input shape matches model expectations
    if X.shape[-1] != FEATURE_COUNT:
        raise ValueError(f"Input feature dimension mismatch. Expected {FEATURE_COUNT}, got {X.shape[-1]}")
    
    return X, y

def prepare_data(df):
    """Prepare training data with shape validation"""
    df = add_technical_indicators(df)
    feature_cols = [col for col in df.columns if col != 'timestamp']
    
    # Scaling
    feature_scaler = RobustScaler()
    close_scaler = RobustScaler()
    
    features = feature_scaler.fit_transform(df[feature_cols])
    close_prices = close_scaler.fit_transform(df[['close']])
    
    # Combine and create sequences
    processed_data = np.concatenate([close_prices, features], axis=1)
    X, y = create_sequences(processed_data, SEQUENCE_LENGTH, FORECAST_HORIZON)
    
    # Train-test split with shuffling
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(len(X) * TRAIN_TEST_SPLIT)
    
    return (
        X[indices[:split]], y[indices[:split]],
        X[indices[split:]], y[indices[split:]],
        feature_scaler, close_scaler, feature_cols
    )

def build_model(hp):
    """Build model with verified input shape"""
    model = Sequential([
        Conv1D(
            filters=hp.Int('conv_filters', 32, 128, step=32),
            kernel_size=hp.Int('kernel_size', 3, 5, step=1),
            activation='relu',
            input_shape=(SEQUENCE_LENGTH, FEATURE_COUNT)
        ),
        MaxPooling1D(2),
        LSTM(
            hp.Int('lstm_units', 32, 128, step=32),
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ),
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
    """Train model with enhanced validation"""
    symbol = f"{token}{FIAT_CURRENCY}"
    tag = f"{token}_{timeframe}"
    model_path = os.path.join(MODEL_DIR, f"{tag}_model.h5")
    
    if os.path.exists(model_path):
        logger.info(f"Skipping {tag} - model exists")
        return
    
    try:
        logger.info(f"Starting training for {tag}")
        send_telegram_message(f"🚀 Starting {tag} training")
        
        df = fetch_binance_data(symbol, timeframe)
        X_train, y_train, X_test, y_test, feat_scaler, close_scaler, _ = prepare_data(df)
        
        # Verify shapes before training
        logger.info(f"Input shape: {X_train.shape}")
        logger.info(f"Output shape: {y_train.shape}")
        
        tuner = kt.Hyperband(
            build_model,
            objective='val_loss',
            max_epochs=MAX_TRAIN_EPOCHS,
            factor=3,
            directory=os.path.join(MODEL_DIR, 'tuning'),
            project_name=tag,
            overwrite=True
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
        
        logger.info(f"Completed training for {tag}")
        send_telegram_message(f"✅ {tag} training complete")
        
    except Exception as e:
        logger.error(f"Failed training {tag}: {str(e)}")
        send_telegram_message(f"❌ {tag} failed: {str(e)}")
        raise

def main():
    """Main training pipeline with error handling"""
    try:
        logger.info("Starting training pipeline")
        send_telegram_message("🏁 Starting training pipeline")
        
        start_time = time.time()
        
        for timeframe in TIMEFRAMES:
            for token in SUPPORTED_TOKENS:
                train_model(token, timeframe)
        
        total_time = (time.time() - start_time) / 3600
        logger.info(f"Training completed in {total_time:.2f} hours")
        send_telegram_message(f"🎉 All models trained in {total_time:.2f} hours")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        send_telegram_message(f"🔥 Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()

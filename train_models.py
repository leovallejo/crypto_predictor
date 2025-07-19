import os
import numpy as np
import pandas as pd
import requests
import joblib
import logging
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, LSTM, Dense, 
                                   Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, 
                                      ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Huber
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import keras_tuner as kt
from config import *

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Limit GPU memory growth to prevent OOM errors
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

# Configure logging
logging.basicConfig(
    filename=os.path.join(MODEL_DIR, 'training.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def send_telegram_message(message):
    """Send notifications via Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        logger.error(f"Telegram error: {str(e)}")

def fetch_binance_data(symbol, interval, limit=DATA_FETCH_LIMIT):
    """Fetch OHLCV data from Binance API"""
    try:
        url = f"https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_av", "trades", "tb_base_av",
            "tb_quote_av", "ignore"
        ])
        
        # Keep only relevant columns
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df.astype(float)
        
    except Exception as e:
        logger.error(f"Error fetching {symbol} data: {str(e)}")
        raise

def calculate_rsi(prices, window=14):
    """Relative Strength Index (Pure Python Implementation)"""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    return 100 - (100 / (1 + rs))

def add_technical_indicators(df):
    """Add all technical indicators to dataframe"""
    # Price transformations
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['close_5ma'] = df['close'].rolling(5).mean()
    
    # Moving Averages
    for ma in TECHNICAL_INDICATORS['MA']:
        df[f'MA_{ma}'] = df['close'].rolling(ma).mean()
    
    # Exponential Moving Averages
    for ema in TECHNICAL_INDICATORS['EMA']:
        df[f'EMA_{ema}'] = df['close'].ewm(span=ema, adjust=False).mean()
    
    # RSI
    for rsi in TECHNICAL_INDICATORS['RSI']:
        df[f'RSI_{rsi}'] = calculate_rsi(df['close'], window=rsi)
    
    # MACD
    if TECHNICAL_INDICATORS['MACD']:
        fast, slow, signal = TECHNICAL_INDICATORS['MACD']
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_SIGNAL'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    
    # Bollinger Bands
    for bb in TECHNICAL_INDICATORS['BOLLINGER']:
        rolling_mean = df['close'].rolling(bb).mean()
        rolling_std = df['close'].rolling(bb).std()
        df[f'BB_UPPER_{bb}'] = rolling_mean + 2 * rolling_std
        df[f'BB_MIDDLE_{bb}'] = rolling_mean
        df[f'BB_LOWER_{bb}'] = rolling_mean - 2 * rolling_std
    
    return df.dropna()

def create_sequences(data, seq_length, horizon):
    """Create time-series sequences for training"""
    X, y = [], []
    for i in range(seq_length, len(data) - horizon):
        X.append(data[i-seq_length:i])
        y.append(data[i:i+horizon, 0])  # Predict only close price
    return np.array(X), np.array(y)

def prepare_data(df):
    """Prepare data for training with proper scaling"""
    # Add indicators
    df = add_technical_indicators(df)
    
    # Select features (all except timestamp)
    feature_cols = [col for col in df.columns if col != 'timestamp']
    
    # Initialize scalers
    feature_scaler = RobustScaler()
    close_scaler = RobustScaler()
    
    # Scale features
    features = feature_scaler.fit_transform(df[feature_cols])
    
    # Scale close prices separately
    close_prices = close_scaler.fit_transform(df[['close']])
    
    # Combine features and close prices
    processed_data = np.concatenate([close_prices, features], axis=1)
    
    # Create sequences
    X, y = create_sequences(processed_data, SEQUENCE_LENGTH, FORECAST_HORIZON)
    
    # Train-test split (time-series aware)
    split = int(len(X) * TRAIN_TEST_SPLIT)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, y_train, X_test, y_test, feature_scaler, close_scaler, feature_cols

def build_model(hp):
    """Build CNN-LSTM model with hyperparameter tuning"""
    model = Sequential()
    
    # Convolutional layers
    model.add(Conv1D(
        filters=hp.Int('conv_filters', 32, 256, step=32),
        kernel_size=hp.Int('kernel_size', 3, 9, step=2),
        activation='relu',
        input_shape=(SEQUENCE_LENGTH, len(FEATURES) + 1),  # +1 for close price
        kernel_regularizer=l2(hp.Float('l2_reg', 1e-5, 1e-3, sampling='log'))
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    
    # LSTM layers
    model.add(LSTM(
        hp.Int('lstm_units', 64, 256, step=32),
        return_sequences=False,
        kernel_regularizer=l2(hp.Float('l2_reg', 1e-5, 1e-3, sampling='log'))
    ))
    model.add(Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1)))
    
    # Output layer
    model.add(Dense(FORECAST_HORIZON))
    
    # Compile with Huber loss
    model.compile(
        optimizer=Adam(hp.Float('lr', 1e-5, 1e-3, sampling='log')),
        loss=Huber(),
        metrics=['mae']
    )
    
    return model

def train_model(token, timeframe):
    """Train model for a specific token and timeframe"""
    symbol = f"{token}{FIAT_CURRENCY}"
    tag = f"{token}_{timeframe}"
    model_path = os.path.join(MODEL_DIR, f"{tag}_model.h5")
    
    # Skip if model already exists
    if os.path.exists(model_path):
        logger.info(f"Skipping {tag} - model exists")
        return
    
    try:
        send_telegram_message(f"🚀 Starting training: {tag}")
        logger.info(f"Training {tag}")
        
        # Fetch and prepare data
        df = fetch_binance_data(symbol, timeframe)
        X_train, y_train, X_test, y_test, feat_scaler, close_scaler, feat_cols = prepare_data(df)
        
        # Initialize tuner
        tuner = kt.Hyperband(
            build_model,
            objective='val_loss',
            max_epochs=MAX_TRAIN_EPOCHS,
            factor=3,
            hyperband_iterations=HYPERBAND_ITERATIONS,
            directory=os.path.join(MODEL_DIR, 'tuning'),
            project_name=tag
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5),
            ModelCheckpoint(model_path, save_best_only=True),
            TensorBoard(log_dir=os.path.join(MODEL_DIR, 'logs', tag))
        ]
        
        # Hyperparameter search
        tuner.search(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save best model and scalers
        best_model = tuner.get_best_models()[0]
        best_model.save(model_path)
        joblib.dump(feat_scaler, os.path.join(MODEL_DIR, f"{tag}_feat_scaler.pkl"))
        joblib.dump(close_scaler, os.path.join(MODEL_DIR, f"{tag}_close_scaler.pkl"))
        
        # Save hyperparameters
        best_hps = tuner.get_best_hyperparameters()[0]
        with open(os.path.join(MODEL_DIR, f"{tag}_hparams.txt"), 'w') as f:
            for k, v in best_hps.values.items():
                f.write(f"{k}: {v}\n")
        
        # Send completion message
        completion_msg = (
            f"✅ {tag} training complete\n"
            f"Best val_loss: {tuner.oracle.get_best_trials()[0].score:.4f}\n"
            f"Parameters: {best_hps.values}"
        )
        send_telegram_message(completion_msg)
        
    except Exception as e:
        error_msg = f"❌ {tag} training failed: {str(e)}"
        logger.error(error_msg)
        send_telegram_message(error_msg)
        raise

def main():
    """Main training function"""
    send_telegram_message("🏁 Starting model training pipeline")
    logger.info("Starting training process")
    
    try:
        for timeframe in TIMEFRAMES:
            for token in SUPPORTED_TOKENS:
                train_model(token, timeframe)
        
        send_telegram_message("🎉 All models trained successfully!")
        logger.info("Training completed successfully")
        
    except Exception as e:
        error_msg = f"🔥 Training failed: {str(e)}"
        logger.error(error_msg)
        send_telegram_message(error_msg)
        raise

if __name__ == "__main__":
    main()

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
import time

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
    # Clear any previous TensorFlow sessions
    tf.keras.backend.clear_session()
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to prevent allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set logical device configuration
            logical_gpus = tf.config.list_logical_devices('GPU')
            logger.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
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
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Telegram error: {e}")

def fetch_binance_data(symbol, interval, limit=DATA_FETCH_LIMIT):
    """Fetch OHLCV data from Binance API with retry logic"""
    max_retries = 5
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            # Add headers to mimic browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/json'
            }
            
            response = requests.get(
                BINANCE_API_URL,
                params=params,
                headers=headers,
                timeout=15
            )
            
            # Check for rate limiting
            if response.status_code == 429:
                wait_time = int(response.headers.get('Retry-After', retry_delay))
                logger.warning(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            
            # Process successful response
            df = pd.DataFrame(
                response.json(),
                columns=["timestamp", "open", "high", "low", "close", "volume",
                        "close_time", "quote_av", "trades", "tb_base_av",
                        "tb_quote_av", "ignore"]
            )
            
            # Select and clean columns
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert to float and handle potential missing values
            df = df.astype(float)
            df = df.dropna()
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed for {symbol}/{interval}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise Exception(f"Failed to fetch data after {max_retries} attempts: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error fetching data: {str(e)}")
            raise

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index with smoothing"""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Use exponential moving average
    avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)  # Add small constant to avoid division by zero
    return 100 - (100 / (1 + rs))

def add_technical_indicators(df):
    """Add comprehensive technical indicators to dataframe"""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")
    
    # Calculate returns and simple moving averages
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['close_5ma'] = df['close'].rolling(5).mean()
    
    # Add moving averages
    for ma in TECHNICAL_INDICATORS['MA']:
        df[f'MA_{ma}'] = df['close'].rolling(ma).mean()
    
    # Add exponential moving averages
    for ema in TECHNICAL_INDICATORS['EMA']:
        df[f'EMA_{ema}'] = df['close'].ewm(span=ema, adjust=False).mean()
    
    # Add RSI
    df['RSI_14'] = calculate_rsi(df['close'])
    
    # Add MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_HIST'] = df['MACD'] - df['MACD_SIGNAL']
    
    # Add Bollinger Bands
    df['BB_MA20'] = df['close'].rolling(20).mean()
    rolling_std = df['close'].rolling(20).std()
    df['BB_UPPER'] = df['BB_MA20'] + 2 * rolling_std
    df['BB_LOWER'] = df['BB_MA20'] - 2 * rolling_std
    df['BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MA20']
    
    # Add Average True Range
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    df['TR'] = np.maximum.reduce([high_low, high_close, low_close])
    df['ATR_14'] = df['TR'].rolling(14).mean()
    
    # Add On-Balance Volume
    df['daily_ret'] = df['close'].pct_change()
    df['OBV'] = (np.sign(df['daily_ret']) * df['volume']).cumsum()
    
    return df.dropna()

def create_sequences(data, seq_length, horizon):
    """Create time-series sequences with input validation"""
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be numpy array")
    
    if len(data.shape) != 2:
        raise ValueError("Data must be 2-dimensional")
    
    if seq_length <= 0 or horizon <= 0:
        raise ValueError("Sequence length and horizon must be positive integers")
    
    X, y = [], []
    for i in range(seq_length, len(data) - horizon):
        X.append(data[i-seq_length:i])
        y.append(data[i:i+horizon, 0])  # Predict close price only
    
    # Convert to numpy arrays with float32 precision
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    return X, y

def prepare_data(df):
    """Prepare training data with robust validation and scaling"""
    # Validate input
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
    
    # Add technical indicators
    df = add_technical_indicators(df)
    feature_cols = [col for col in df.columns if col != 'timestamp']
    
    # Initialize scalers
    feature_scaler = RobustScaler()
    close_scaler = RobustScaler()
    
    # Scale features
    features = feature_scaler.fit_transform(df[feature_cols])
    close_prices = close_scaler.fit_transform(df[['close']])
    
    # Combine features and targets
    processed_data = np.concatenate([close_prices, features], axis=1)
    
    # Create sequences
    X, y = create_sequences(processed_data, SEQUENCE_LENGTH, FORECAST_HORIZON)
    
    # Train-test split with shuffling
    split_idx = int(len(X) * TRAIN_TEST_SPLIT)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    X_train = X[indices[:split_idx]]
    y_train = y[indices[:split_idx]]
    X_test = X[indices[split_idx:]]
    y_test = y[indices[split_idx:]]
    
    return X_train, y_train, X_test, y_test, feature_scaler, close_scaler, feature_cols

def build_model(hp):
    """Build optimized CNN-LSTM model with hyperparameter tuning"""
    model = Sequential()
    
    # Convolutional layers
    model.add(Conv1D(
        filters=hp.Int('conv_filters', 32, 256, step=32),
        kernel_size=hp.Int('kernel_size', 3, 9, step=2),
        activation='relu',
        input_shape=(SEQUENCE_LENGTH, len(FEATURES) + 1),
        kernel_regularizer=l2(hp.Float('l2_reg_conv', 1e-5, 1e-3)),
        padding='same'
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    # LSTM layers
    model.add(LSTM(
        units=hp.Int('lstm_units', 64, 256, step=32),
        return_sequences=False,
        kernel_regularizer=l2(hp.Float('l2_reg_lstm', 1e-5, 1e-3))
    ))
    model.add(Dropout(hp.Float('dropout', 0.2, 0.5)))
    
    # Dense layers
    model.add(Dense(
        units=hp.Int('dense_units', 32, 128, step=32),
        activation='relu',
        kernel_regularizer=l2(hp.Float('l2_reg_dense', 1e-5, 1e-3))
    ))
    model.add(Dense(FORECAST_HORIZON))
    
    # Compile model
    optimizer = Adam(
        learning_rate=hp.Float('lr', 1e-5, 1e-3, sampling='log'),
        clipvalue=0.5
    )
    
    model.compile(
        optimizer=optimizer,
        loss=Huber(),
        metrics=['mae', 'mse']
    )
    
    return model

def train_model(token, timeframe):
    """Train model for specific token/timeframe with comprehensive logging"""
    symbol = f"{token}{FIAT_CURRENCY}"
    tag = f"{token}_{timeframe}"
    model_path = os.path.join(MODEL_DIR, f"{tag}_model.h5")
    
    # Check for existing model
    if os.path.exists(model_path):
        logger.info(f"Model {tag} already exists, skipping training")
        return
    
    try:
        start_time = time.time()
        logger.info(f"Starting training for {tag}")
        send_telegram_message(f"🚀 Starting training for {tag}")
        
        # Fetch and prepare data
        df = fetch_binance_data(symbol, timeframe)
        X_train, y_train, X_test, y_test, feat_scaler, close_scaler, feat_cols = prepare_data(df)
        
        # Initialize tuner
        tuner = kt.Hyperband(
            build_model,
            objective='val_loss',
            max_epochs=MAX_TRAIN_EPOCHS,
            factor=3,
            directory=os.path.join(MODEL_DIR, 'tuning'),
            project_name=tag,
            overwrite=True,
            seed=42
        )
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                mode='min',
                verbose=1
            ),
            TensorBoard(
                log_dir=os.path.join(MODEL_DIR, 'logs', tag),
                histogram_freq=1
            )
        ]
        
        # Perform hyperparameter search
        logger.info(f"Starting hyperparameter search for {tag}")
        tuner.search(
            X_train, y_train,
            epochs=MAX_TRAIN_EPOCHS,
            validation_data=(X_test, y_test),
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Get best model
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.get_best_models(num_models=1)[0]
        
        # Save model and scalers
        best_model.save(model_path)
        joblib.dump(feat_scaler, os.path.join(MODEL_DIR, f"{tag}_feat_scaler.pkl"))
        joblib.dump(close_scaler, os.path.join(MODEL_DIR, f"{tag}_close_scaler.pkl"))
        
        # Log results
        training_time = (time.time() - start_time) / 60
        best_val_loss = tuner.oracle.get_best_trials(num_trials=1)[0].score
        logger.info(f"Training completed for {tag} | Time: {training_time:.2f} mins | Val Loss: {best_val_loss:.4f}")
        send_telegram_message(
            f"✅ Training completed for {tag}\n"
            f"⏱ Time: {training_time:.2f} mins\n"
            f"📉 Val Loss: {best_val_loss:.4f}\n"
            f"🔍 Best Hyperparams: {best_hps.values}"
        )
        
    except Exception as e:
        error_msg = f"❌ Training failed for {tag}: {str(e)}"
        logger.error(error_msg)
        send_telegram_message(error_msg)
        raise

def main():
    """Main training pipeline with error handling"""
    try:
        logger.info("Starting training pipeline")
        send_telegram_message("🏁 Starting training pipeline")
        
        start_time = time.time()
        
        # Train models for all token/timeframe combinations
        for timeframe in TIMEFRAMES:
            for token in SUPPORTED_TOKENS:
                train_model(token, timeframe)
        
        # Calculate total training time
        total_time = (time.time() - start_time) / 3600  # in hours
        completion_msg = f"🎉 All models trained successfully | Total Time: {total_time:.2f} hours"
        logger.info(completion_msg)
        send_telegram_message(completion_msg)
        
    except Exception as e:
        error_msg = f"🔥 Critical error in training pipeline: {str(e)}"
        logger.error(error_msg)
        send_telegram_message(error_msg)
        raise

if __name__ == "__main__":
    main()

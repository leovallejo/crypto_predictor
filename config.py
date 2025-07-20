import os
from dotenv import load_dotenv

load_dotenv()

# Directory Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Trading Config
SUPPORTED_TOKENS = ['BTC', 'ETH']
TIMEFRAMES = ['1h', '4h']
FIAT_CURRENCY = 'USDT'

# Model Config
SEQUENCE_LENGTH = 60
FORECAST_HORIZON = 7
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
MAX_TRAIN_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
HYPERBAND_ITERATIONS = 1
BATCH_SIZE = 32

# Feature Engineering - Updated to match actual feature count
TECHNICAL_INDICATORS = {
    'MA': [7, 14],    # 2 features
    'RSI': True,      # 1 feature
    'MACD': True      # 2 features (MACD and Signal line)
}

# Base features (open, high, low, close, volume, log_ret) = 6 features
# MA: 2, RSI: 1, MACD: 2 = 5 additional features
FEATURE_COUNT = 6 + 2 + 1 + 2  # Total 11 features

# API Config
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
DATA_FETCH_LIMIT = 500

# Telegram Config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Logging Config
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_FILE = os.path.join(BASE_DIR, 'training.log')

# Live Monitor Config
REFRESH_INTERVAL = 5
SHOW_LAST_N = 10
PREDICTION_HISTORY_FILE = os.path.join(DATA_DIR, 'prediction_history.csv')

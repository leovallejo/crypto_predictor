import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Directory Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Trading Config (Reduced for CPU)
SUPPORTED_TOKENS = ['BTC', 'ETH']  # Fewer tokens
TIMEFRAMES = ['1h', '4h']         # Fewer timeframes
FIAT_CURRENCY = 'USDT'

# Model Config (Optimized for CPU)
SEQUENCE_LENGTH = 60              # Reduced from 90
FORECAST_HORIZON = 7              # Reduced from 14
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
MAX_TRAIN_EPOCHS = 50             # Reduced from 100
EARLY_STOPPING_PATIENCE = 5       # Reduced from 10
HYPERBAND_ITERATIONS = 1          # Reduced from 2
BATCH_SIZE = 32                   # Reduced from 64

# Feature Engineering (Simplified)
TECHNICAL_INDICATORS = {
    'MA': [7, 14],
    'EMA': [9, 21],
    'RSI': [14],
    'MACD': [12, 26, 9]
}

# API Config
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
DATA_FETCH_LIMIT = 1000           # Reduced from 2000

# Telegram Config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Logging Config
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_FILE = os.path.join(BASE_DIR, 'training.log')

# Live Monitor Config
REFRESH_INTERVAL = 5              # Increased from 2
SHOW_LAST_N = 10                  # Reduced from 15
PREDICTION_HISTORY_FILE = os.path.join(DATA_DIR, 'prediction_history.csv')

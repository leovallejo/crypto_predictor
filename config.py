import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Directory Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# --- Trading Config ---
SUPPORTED_TOKENS = ['BTC', 'ETH', 'BNB', 'SOL', 'ARB']
TIMEFRAMES = ['1h', '4h', '1d']
FIAT_CURRENCY = 'USDT'

# --- Model Config ---
SEQUENCE_LENGTH = 90
FORECAST_HORIZON = 14
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
MAX_TRAIN_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
HYPERBAND_ITERATIONS = 2
BATCH_SIZE = 64

# --- Feature Engineering ---
TECHNICAL_INDICATORS = {
    'MA': [7, 14, 21],
    'EMA': [9, 21, 50],
    'RSI': [14],
    'MACD': [12, 26, 9],
    'BOLLINGER': [20],
    'ATR': [14],
    'OBV': [],
    'STOCH': [14, 3, 3]
}

# --- API Config ---
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
DATA_FETCH_LIMIT = 2000

# --- Telegram ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# --- Logging ---
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_FILE = os.path.join(BASE_DIR, 'training.log')

# --- Live Monitor ---
REFRESH_INTERVAL = 2
SHOW_LAST_N = 15
PREDICTION_HISTORY_FILE = os.path.join(DATA_DIR, 'prediction_history.csv')

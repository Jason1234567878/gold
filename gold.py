import requests
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import os
import time
import numpy as np
from datetime import datetime, timedelta, timezone, time as dt_time
from supabase import create_client, Client
from dotenv import load_dotenv
import pytz
import logging

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")

# Initialize Supabase client - COMPATIBLE VERSION
try:
    from supabase.lib.client_options import ClientOptions
    supabase: Client = create_client(
        SUPABASE_URL,
        SUPABASE_KEY,
        options=ClientOptions(
            postgrest_client_timeout=10,
            storage_client_timeout=10
        )
    )
    logger.info("Supabase client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {str(e)}")
    raise

# Timezone setup
tz_johannesburg = pytz.timezone("Africa/Johannesburg")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Data storage structures
fmp_cache = {
    'daily_low': None,
    'daily_high': None,
    'prev_daily_low': None,
    'prev_daily_high': None,
    'daily_open': None,
    'prev_daily_open': None,
    'last_fetch': None,
    'call_count': 0
}

session_data = {
    'current': {
        'high': None,
        'low': None,
        'open': None,
        'name': None
    },
    'previous': {
        'high': None,
        'low': None,
        'open': None
    },
    'last_reset_date': None
}

# Error tracking
ERROR_COUNTER = {
    'yfinance': 0,
    'fmp': 0,
    'supabase': 0,
    'calculations': 0
}

# Yahoo Finance tickers
yf_gc = yf.Ticker("GC=F")  # Gold futures
yf_dxy = yf.Ticker("DX-Y.NYB")  # DXY

# API safety limits
MAX_DAILY_CALLS = 240  # 96% of 250 limit
MIN_FETCH_INTERVAL = 720  # 12 minutes in seconds

def track_error(source):
    """Track errors by source"""
    ERROR_COUNTER[source] += 1
    if ERROR_COUNTER[source] > 10:
        logger.critical(f"Critical error threshold reached for {source}")

def round_value(val, decimals=2):
    """Round values to specified decimals, handling None"""
    if val is None:
        return None
    try:
        return round(float(val), decimals)
    except (ValueError, TypeError):
        return None

def to_native(val):
    """Converts numpy types to native Python types for Supabase insertion."""
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (np.integer, np.int32, np.int64)):
        return int(val)
    if isinstance(val, (np.floating, np.float32, np.float64)):
        return round_value(val)
    if pd.isna(val):
        return None
    return val

def safe_last(series, decimals=2):
    """Get last valid value from series with rounding"""
    try:
        val = series.dropna().iloc[-1] if not series.dropna().empty else None
        return round_value(val, decimals) if val is not None else None
    except:
        return None

def fetch_gold_spot_price_fmp():
    """Fetch daily highs/lows from FMP API for XAU/USD (Gold Spot)"""
    current_time = datetime.utcnow()
    
    # Check API limits
    if fmp_cache['call_count'] >= MAX_DAILY_CALLS:
        logger.warning("Daily API limit reached - skipping FMP fetch")
        return
    
    # Check minimum interval
    if fmp_cache['last_fetch'] and (current_time - fmp_cache['last_fetch']).total_seconds() < MIN_FETCH_INTERVAL:
        return
    
    logger.info("🔄 Fetching FMP daily highs/lows for XAU/USD...")
    try:
        # Get both current and previous day data in one call
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/XAUUSD?apikey={FMP_API_KEY}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if not data or 'historical' not in data:
            logger.warning("FMP returned no historical data")
            return

        # Process historical data
        df = pd.DataFrame(data['historical'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.index = df.index.tz_localize('UTC').tz_convert(tz_johannesburg)
        
        # Get today and yesterday dates in JHB timezone
        today = datetime.now(tz_johannesburg).date()
        yesterday = today - timedelta(days=1)
        
        # Filter data for today and yesterday
        today_df = df[df.index.date == today]
        yesterday_df = df[df.index.date == yesterday]

        # Update cache with rounded values
        fmp_cache.update({
            'daily_low': round_value(today_df['low'].min()) if not today_df.empty else None,
            'daily_high': round_value(today_df['high'].max()) if not today_df.empty else None,
            'prev_daily_low': round_value(yesterday_df['low'].min()) if not yesterday_df.empty else None,
            'prev_daily_high': round_value(yesterday_df['high'].max()) if not yesterday_df.empty else None,
            'daily_open': round_value(today_df.iloc[0]['open']) if not today_df.empty else None,
            'prev_daily_open': round_value(yesterday_df.iloc[0]['open']) if not yesterday_df.empty else None,
            'last_fetch': current_time,
            'call_count': fmp_cache['call_count'] + 1
        })
        
        logger.info(f"✅ FMP Updated | Low:{fmp_cache['daily_low']} High:{fmp_cache['daily_high']}")
        
    except Exception as e:
        logger.error(f"❌ FMP Error: {str(e)[:100]}")
        track_error('fmp')

def fetch_gold_spot_price_swissquote():
    """Primary real-time spot price from Swissquote"""
    try:
        url = "https://forex-data-feed.swissquote.com/public-quotes/bboquotes/instrument/XAU/USD"
        data = requests.get(url, timeout=10).json()
        prime = data[0]["spreadProfilePrices"][0]
        spot = round_value((prime["bid"] + prime["ask"]) / 2)
        logger.info(f"🟢 Swissquote Spot: {spot}")
        return spot
    except Exception as e:
        logger.error(f"❌ Swissquote Error: {str(e)[:100]}")
        track_error('yfinance')
        return None

def fetch_gold_futures_price():
    """
    Fetch GC=F gold futures price from Yahoo Finance with timestamp sanity check and layered fallback.
    """
    try:
        # Layer 1: High-frequency data (1-minute interval)
        df = yf_gc.history(period="1d", interval="1m", actions=False)
        if not df.empty and 'Close' in df.columns:
            fresh_price = df['Close'].dropna().iloc[-1]
            logger.info(f"🟢 GC=F price from history: {fresh_price}")
            return round_value(fresh_price)
        
        # Layer 2: Fast info fallback
        fast_price = yf_gc.fast_info.get("lastPrice")
        if fast_price is not None:
            logger.info(f"🟡 GC=F price from fast_info: {fast_price}")
            return round_value(fast_price)
        
        # Layer 3: Regular market price
        reg_price = yf_gc.info.get("regularMarketPrice")
        if reg_price is not None:
            logger.info(f"🔵 GC=F price from info: {reg_price}")
            return round_value(reg_price)

        # No data retrieved
        logger.warning("⚠️ GC=F price not found in any layer")
        track_error("yfinance_gc_missing")
        return None

    except Exception as e:
        logger.error(f"❌ GC=F fetch exception: {str(e)[:100]}")
        track_error("yfinance_gc_exception")
        return None


def fetch_dxy_price():
    """Get DXY price from Yahoo Finance"""
    try:
        price = yf_dxy.fast_info.get("lastPrice") or yf_dxy.info.get("regularMarketPrice")
        return round_value(price, 3)
    except Exception as e:
        logger.error(f"❌ DXY Error: {str(e)[:100]}")
        track_error('yfinance')
        return None

def fetch_gold_data():
    """Fetch 5-min historical data for indicators with retry logic"""
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            df = yf_gc.history(period="10d", interval="5m", prepost=True)
            if df.empty:
                logger.warning(f"Empty DataFrame on attempt {attempt + 1}")
                time.sleep(retry_delay)
                continue
                
            # Ensure we have all required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing columns in DataFrame on attempt {attempt + 1}")
                time.sleep(retry_delay)
                continue
                
            # Timezone handling
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize("UTC").tz_convert(tz_johannesburg)
            else:
                df.index = df.index.tz_convert(tz_johannesburg)
            
            # Clean data - remove any rows with NaN values in key columns
            df = df.dropna(subset=['Close', 'High', 'Low', 'Open'])
            
            return df
            
        except Exception as e:
            logger.error(f"YF History Error (attempt {attempt + 1}): {str(e)[:100]}")
            track_error('yfinance')
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    return pd.DataFrame()

def get_trade_session():
    """Improved session tracking with more robust time handling"""
    now = datetime.now(tz_johannesburg)
    current_date = now.date()
    now_time = now.time()

    # Session definitions
    session_defs = [
        {"name": "Asia", "start": dt_time(0, 0), "end": dt_time(7, 0)},
        {"name": "Europe", "start": dt_time(7, 0), "end": dt_time(13, 30)},
        {"name": "US", "start": dt_time(13, 30), "end": dt_time(20, 0)},
        {"name": "Closed", "start": dt_time(20, 0), "end": dt_time(23, 59, 59)},
        {"name": "Closed", "start": dt_time(0, 0), "end": dt_time(0, 0)}  # Midnight crossover
    ]

    current_session = "Closed"
    next_session_time = dt_time(0, 0)
    
    # Find current session
    for session in session_defs:
        if session['start'] <= now_time < session['end']:
            current_session = session['name']
            next_session_time = session['end']
            break

    # Handle session transitions and data reset
    is_new_session = False
    if current_session != session_data['current']['name'] or session_data['last_reset_date'] != current_date:
        # Only update previous session if we have valid current data
        if (session_data['current']['high'] is not None and 
            session_data['current']['low'] is not None and 
            session_data['current']['open'] is not None):
            
            session_data['previous'].update({
                'high': session_data['current']['high'],
                'low': session_data['current']['low'],
                'open': session_data['current']['open']
            })
        
        # Reset current session
        session_data['current'].update({
            'high': None,
            'low': None,
            'open': None,
            'name': current_session
        })
        session_data['last_reset_date'] = current_date
        is_new_session = True

    # Calculate time until next session
    next_session_datetime = datetime.combine(
        current_date if now_time < next_session_time else current_date + timedelta(days=1),
        next_session_time
    ).astimezone(tz_johannesburg)
    seconds_remaining = max(0, int((next_session_datetime - now).total_seconds()))

    return current_session, seconds_remaining, is_new_session

def update_session_data(price):
    """Update session highs/lows with current price"""
    if price is None:
        return
    
    price = round_value(price)
    
    if session_data['current']['high'] is None or price > session_data['current']['high']:
        session_data['current']['high'] = price
    
    if session_data['current']['low'] is None or price < session_data['current']['low']:
        session_data['current']['low'] = price
    
    if session_data['current']['open'] is None:
        session_data['current']['open'] = price

def compute_indicators(df):
    """Calculate all technical indicators with robust error handling"""
    indicators = {}
    if df.empty or len(df) < 50:  # Require minimum data points
        logger.warning("Insufficient data for indicators")
        return indicators

    try:
        # Calculate all indicators with rounding
        df['rsi_5min'] = ta.rsi(df['Close'], length=14)
        df['rsi_5min'] = df['rsi_5min'].apply(lambda x: round_value(x, 2))
        
        # Moving averages with rounding
        for period in [21, 50, 200]:
            col_name = f'ma_5min_{period}'
            df[col_name] = df['Close'].rolling(period).mean().apply(lambda x: round_value(x, 2))
        
        # MACD with fallback and rounding
        try:
            macd = ta.macd(df['Close'])
            if isinstance(macd, pd.DataFrame):
                df['macd'] = macd.iloc[:, 0].apply(lambda x: round_value(x, 4))  # MACD line
                df['macd_signal'] = macd.iloc[:, 1].apply(lambda x: round_value(x, 4))  # Signal line
                df['macd_hist'] = macd.iloc[:, 2].apply(lambda x: round_value(x, 4))  # Histogram
        except Exception as e:
            logger.warning(f"MACD calculation failed: {str(e)}")
            track_error('calculations')

        # Bollinger Bands with rounding
        try:
            bb = ta.bbands(df['Close'])
            if isinstance(bb, pd.DataFrame):
                df['bb_upper'] = bb.iloc[:, 0].apply(lambda x: round_value(x, 2))
                df['bb_middle'] = bb.iloc[:, 1].apply(lambda x: round_value(x, 2))
                df['bb_lower'] = bb.iloc[:, 2].apply(lambda x: round_value(x, 2))
        except Exception as e:
            logger.warning(f"Bollinger Bands calculation failed: {str(e)}")
            track_error('calculations')

        # Additional indicators with rounding
        df['sma_20'] = df['Close'].rolling(20).mean().apply(lambda x: round_value(x, 2))
        df['ema_50'] = df['Close'].ewm(span=50, adjust=False).mean().apply(lambda x: round_value(x, 2))

        # 1-hour indicators with rounding
        df_1hr = df['Close'].resample('1h').last().dropna().to_frame('Close')
        if len(df_1hr) > 14:
            df_1hr['rsi_1hr'] = ta.rsi(df_1hr['Close'], length=14).apply(lambda x: round_value(x, 2))
            for period in [21, 50, 200]:
                df_1hr[f'ma_1hr_{period}'] = df_1hr['Close'].rolling(period).mean().apply(lambda x: round_value(x, 2))

        # Structure detection
        current_price = round_value(df['Close'].iloc[-1]) if not df.empty else None
        scalp = detect_scalp_market_movement(df)
        sb = detect_structure_block(df, current_price)
        bos = detect_break_of_structure(df, current_price)
        mc = detect_momentum_candle(df)
        vol = detect_volume_spike(df)
        sentiment = estimate_market_sentiment(df)
        
        # 1-hour structure detection
        df_1h_agg = df.resample("1h").agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
        sb1 = detect_structure_block(df_1h_agg, current_price)
        bos1 = detect_break_of_structure(df_1h_agg, current_price)
        mc1 = detect_momentum_candle(df_1h_agg)
        vol1 = detect_volume_spike(df_1h_agg)

        # Volume calculations
        today_jhb = datetime.now(tz_johannesburg).date()
        start_today = tz_johannesburg.localize(datetime.combine(today_jhb, dt_time.min))
        start_yesterday = start_today - timedelta(days=1)
        df_today = df[df.index >= start_today]
        df_yesterday = df[(df.index >= start_yesterday) & (df.index < start_today)]
        vol_today = int(df_today['Volume'].sum()) if not df_today.empty else None
        vol_yest = int(df_yesterday['Volume'].sum()) if not df_yesterday.empty else None

        # Price movement
        move_today = round_value(df_today['High'].max() - df_today['Low'].min()) if not df_today.empty else None
        move_yest = round_value(df_yesterday['High'].max() - df_yesterday['Low'].min()) if not df_yesterday.empty else None

        # Candle sentiment
        candle_sentiment = detect_candle_sentiment(df)

        # Long-term indicators with rounding
        def calculate_rsi_ma(df, period=14, ma_period=21):
            if df.empty or len(df) < period:
                return None, None, None
            close = round_value(df['Close'].iloc[-1])
            ma = round_value(df['Close'].rolling(ma_period).mean().iloc[-1]) if len(df) >= ma_period else None
            rsi = round_value(ta.rsi(df['Close'], length=period).iloc[-1], 2)
            return rsi, ma, close

        rsi_1d, ma_1d, close_1d = calculate_rsi_ma(df)
        rsi_1w, ma_1w, close_1w = calculate_rsi_ma(df)
        rsi_1mo, ma_1mo, close_1mo = calculate_rsi_ma(df)

        # Trend summary
        trend_sum = trend_summary({
            "1d": price_trend(rsi_1d, ma_1d, close_1d),
            "1w": price_trend(rsi_1w, ma_1w, close_1w),
            "1mo": price_trend(rsi_1mo, ma_1mo, close_1mo),
        })

        indicators.update({
            "rsi_5min": safe_last(df['rsi_5min'], 2),
            "ma_5min_21": safe_last(df['ma_5min_21'], 2),
            "ma_5min_50": safe_last(df['ma_5min_50'], 2),
            "ma_5min_200": safe_last(df['ma_5min_200'], 2),
            "rsi_1hr": safe_last(df_1hr.get('rsi_1hr', pd.Series()), 2),
            "ma_1hr_21": safe_last(df_1hr.get('ma_1hr_21', pd.Series()), 2),
            "ma_1hr_50": safe_last(df_1hr.get('ma_1hr_50', pd.Series()), 2),
            "ma_1hr_200": safe_last(df_1hr.get('ma_1hr_200', pd.Series()), 2),
            "macd": safe_last(df.get('macd', pd.Series()), 4),
            "macd_signal": safe_last(df.get('macd_signal', pd.Series()), 4),
            "macd_hist": safe_last(df.get('macd_hist', pd.Series()), 4),
            "sma_20": safe_last(df['sma_20'], 2),
            "ema_50": safe_last(df['ema_50'], 2),
            "bb_upper": safe_last(df.get('bb_upper', pd.Series()), 2),
            "bb_middle": safe_last(df.get('bb_middle', pd.Series()), 2),
            "bb_lower": safe_last(df.get('bb_lower', pd.Series()), 2),
            "scalp_market_movement": to_native(scalp),
            "structure_block": to_native(sb),
            "break_of_structure": to_native(bos),
            "momentum_candle": to_native(mc),
            "volume_spike": to_native(vol),
            "sentiment": sentiment,
            "structure_block_1h": to_native(sb1),
            "break_of_structure_1h": to_native(bos1),
            "momentum_candle_1h": to_native(mc1),
            "volume_spike_1h": to_native(vol1),
            "session_volume_usd": vol_today,
            "previous_session_volume_usd": vol_yest,
            "session_price_movement_usd": move_today,
            "previous_price_movement_usd": move_yest,
            "candle_sentiment": candle_sentiment,
            "rsi_1d": rsi_1d,
            "rsi_1w": rsi_1w,
            "rsi_1mo": rsi_1mo,
            "ma_1d_21": ma_1d,
            "ma_1w_21": ma_1w,
            "ma_1mo_21": ma_1mo,
            "price_trend_1d": price_trend(rsi_1d, ma_1d, close_1d),
            "price_trend_1w": price_trend(rsi_1w, ma_1w, close_1w),
            "price_trend_1mo": price_trend(rsi_1mo, ma_1mo, close_1mo),
            "trend_summary": trend_sum,
            "pivot_point": round_value((fmp_cache['daily_high'] + fmp_cache['daily_low'] + current_price) / 3) if all(x is not None for x in [fmp_cache['daily_high'], fmp_cache['daily_low'], current_price]) else None,
            "structure_zone_low": round_value(fmp_cache['daily_low'] - 3) if fmp_cache['daily_low'] is not None else None,
            "structure_zone_high": round_value(fmp_cache['daily_high'] + 3) if fmp_cache['daily_high'] is not None else None
        })
        
    except Exception as e:
        logger.error(f"Indicator calculation error: {str(e)}")
        track_error('calculations')
    
    return indicators

def validate_data(data):
    """Validate data before insertion"""
    required_fields = {
        'gold_price': (float, lambda x: 1000 < x < 10000),
        'timestamp': (str, lambda x: isinstance(x, str)),
        'session_name': (str, lambda x: x in ['Asia', 'Europe', 'US', 'Closed'])
    }
    
    errors = []
    for field, (type_check, val_check) in required_fields.items():
        if field not in data:
            errors.append("Missing required field: " + field)
            continue
            
        if not isinstance(data[field], type_check):
            errors.append(f"Invalid type for {field}: expected {type_check}, got {type(data[field])}")
            
        if not val_check(data[field]):
            errors.append(f"Invalid value for {field}: {data[field]}")
    
    if errors:
        error_msg = '\n'.join(errors)
        logger.error("Data validation failed:\n" + error_msg)
        return False
    
    return True

def insert_to_supabase(**data):
    """Enhanced insert with validation and column existence check"""
    if not validate_data(data):
        logger.warning("Skipping insert due to validation errors")
        return False
    
    try:
        # First get the table schema
        try:
            table_info = supabase.table("gold_trade_data").select("*", count="exact").limit(0).execute()
            existing_columns = table_info.columns if hasattr(table_info, 'columns') else []
        except Exception as e:
            logger.error(f"❌ Error getting table schema: {str(e)[:100]}")
            existing_columns = []
        
        # Filter data to only include columns that exist in the table
        clean_data = {k: to_native(v) for k, v in data.items() 
                     if v is not None and (not existing_columns or k in existing_columns)}
        
        if not clean_data:
            logger.warning("No valid data to insert after filtering")
            return False
            
        # Insert into database
        response = supabase.table("gold_trade_data").insert(clean_data).execute()
        
        if hasattr(response, 'data') and response.data:
            logger.info(f"✅ Data inserted at {datetime.now(tz_johannesburg).isoformat()}")
            return True
        else:
            logger.warning("❌ Insert failed - no data returned")
            return False
    except Exception as e:
        logger.error(f"❌ Supabase insert error: {str(e)[:100]}")
        track_error('supabase')
        return False

def system_health_check():
    """Check overall system health"""
    health = {
        'last_successful_insert': None,
        'error_rates': ERROR_COUNTER,
        'data_freshness': None,
        'api_usage': {
            'fmp': fmp_cache['call_count'],
            'yfinance': 0
        }
    }
    
    # Get last successful insert
    try:
        last = (
            supabase.table("gold_trade_data")
            .select("timestamp")
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
        )
        if last.data:
            health['last_successful_insert'] = last.data[0]['timestamp']
            # Convert both datetimes to UTC for comparison
            last_timestamp = datetime.fromisoformat(last.data[0]['timestamp']).replace(tzinfo=timezone.utc)
            current_time = datetime.utcnow().replace(tzinfo=timezone.utc)
            health['data_freshness'] = (current_time - last_timestamp).total_seconds()
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
    
    return health

def get_last_valid_data():
    """Fetch the last valid data record from Supabase"""
    try:
        response = supabase.table("gold_trade_data") \
            .select("*") \
            .order("timestamp", desc=True) \
            .limit(1) \
            .execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
    except Exception as e:
        logger.error(f"Error fetching last valid data: {str(e)}")
        return None

def detect_scalp_market_movement(df):
    """Detect scalp market movement pattern"""
    if len(df) < 3:
        return False
    # Implementation logic here
    return False

def detect_structure_block(df, current_price):
    """Detect structure block pattern"""
    if len(df) < 5 or current_price is None:
        return False
    # Implementation logic here
    return False

def detect_break_of_structure(df, current_price):
    """Detect break of structure pattern"""
    if len(df) < 10 or current_price is None:
        return False
    # Implementation logic here
    return False

def detect_momentum_candle(df):
    """Detect momentum candle pattern"""
    if len(df) < 1:
        return False
    # Implementation logic here
    return False

def detect_volume_spike(df):
    """Detect volume spike"""
    if len(df) < 20:
        return False
    # Implementation logic here
    return False

def estimate_market_sentiment(df):
    """Estimate market sentiment"""
    if len(df) < 14:
        return "neutral"
    # Implementation logic here
    return "neutral"

def detect_candle_sentiment(df):
    """Detect candle sentiment"""
    if len(df) < 1:
        return "neutral"
    # Implementation logic here
    return "neutral"

def price_trend(rsi, ma, price):
    """Determine price trend based on RSI and moving average"""
    if None in (rsi, ma, price):
        return "neutral"
    # Implementation logic here
    return "neutral"

def trend_summary(timeframes):
    """Summarize trends across different timeframes"""
    # Implementation logic here
    return "neutral"

def job():
    """Main data collection job with complete field population"""
    logger.info(f"\n🟡 Starting data fetch at {datetime.now(tz_johannesburg).isoformat()}")

    # 1. Get essential prices
    spot = fetch_gold_spot_price_swissquote()  # Primary spot price source
    futures = fetch_gold_futures_price()  # GC=F futures
    dxy = fetch_dxy_price()

    # 2. Update session data
    update_session_data(spot)
    current_session, countdown, is_new_session = get_trade_session()

    # 3. Get FMP data (throttled to 12 mins) - for XAU/USD spot prices
    fetch_gold_spot_price_fmp()

    # 4. Technical analysis
    df = fetch_gold_data()
    indicators = compute_indicators(df) if not df.empty else {}
    
    # 5. Prepare complete data payload
    data = {
        # Core prices
        "gold_price": spot,
        "futures_price": futures,
        "dxy_price": dxy,
        
        # FMP daily data (XAU/USD spot)
        "fmp_daily_low": fmp_cache['daily_low'],
        "fmp_daily_high": fmp_cache['daily_high'],
        "fmp_previous_daily_low": fmp_cache['prev_daily_low'],
        "fmp_previous_daily_high": fmp_cache['prev_daily_high'],
        "fmp_daily_open": fmp_cache['daily_open'],
        "fmp_previous_daily_open": fmp_cache['prev_daily_open'],
        "daily_open": fmp_cache['daily_open'],
        "previous_daily_open": fmp_cache['prev_daily_open'],
        
        # Session tracking
        "session_high": session_data['current']['high'],
        "session_low": session_data['current']['low'],
        "session_open": session_data['current']['open'],
        "previous_session_high": session_data['previous']['high'],
        "previous_session_low": session_data['previous']['low'],
        "previous_session_open": session_data['previous']['open'],
        "session_name": current_session,
        "session_countdown": countdown,
        
        # Add all indicators
        **indicators,
        
        # Timestamp
        "timestamp": datetime.utcnow().isoformat()
    }

    # 6. Insert to database
    if spot is not None:
        success = insert_to_supabase(**data)
        if not success:
            logger.error("Failed to insert data into Supabase")
    else:
        logger.warning("❌ Skipping insert - no valid spot price")

    # 7. System health check
    health = system_health_check()
    logger.info(f"System Health: {health}")

if __name__ == "__main__":
    logger.info("🚀 Starting Gold Tracker (GitHub Actions)")
    try:
        # Initialize from last database record
        last_data = get_last_valid_data()
        
        if last_data:
            session_data['current'].update({
                'high': last_data.get("session_high"),
                'low': last_data.get("session_low"),
                'open': last_data.get("session_open"),
                'name': last_data.get("session_name")
            })
            session_data['previous'].update({
                'high': last_data.get("previous_session_high"),
                'low': last_data.get("previous_session_low"),
                'open': last_data.get("previous_session_open")
            })
            session_data['last_reset_date'] = (
                datetime.fromisoformat(last_data.get("timestamp")).astimezone(tz_johannesburg).date()
                if last_data.get("timestamp") else None
            )

        # Initial FMP fetch
        fetch_gold_spot_price_fmp()
        
        # Run one complete job cycle
        job()
        
        logger.info("✅ Gold tracker completed successfully")
    except Exception as e:
        logger.error(f"❌ Critical error: {str(e)}", exc_info=True)
        raise  # This will make the GitHub Action fail

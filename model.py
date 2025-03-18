import json
import os
import pickle
from zipfile import ZipFile
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.svm import SVR
import xgboost as xgb
from updater import download_binance_daily_data, download_binance_current_day_data, download_coingecko_data, download_coingecko_current_day_data
from config import data_base_path, model_file_path, TOKEN, MODEL, CG_API_KEY, TRAINING_DAYS, REGION

binance_data_path = os.path.join(data_base_path, "binance")
coingecko_data_path = os.path.join(data_base_path, "coingecko")
training_price_data_path = os.path.join(data_base_path, "price_data.csv")

def download_data_binance(token, training_days, region):
    files = download_binance_daily_data(f"{token}USDT", training_days, region, binance_data_path)
    print(f"Downloaded {len(files)} new files")
    return files

def download_data_coingecko(token, training_days):
    files = download_coingecko_data(token, training_days, coingecko_data_path, CG_API_KEY)
    print(f"Downloaded {len(files)} new files")
    return files

def download_data(token, training_days, region, data_provider):
    if data_provider == "coingecko":
        return download_data_coingecko(token, int(training_days))
    elif data_provider == "binance":
        return download_data_binance(token, training_days, region)
    else:
        raise ValueError("Unsupported data provider")

def format_data(files, data_provider):
    if not files:
        print("Already up to date")
        return
    
    if data_provider == "binance":
        files = sorted([x for x in os.listdir(binance_data_path) if x.startswith(f"{TOKEN}USDT")])
    elif data_provider == "coingecko":
        files = sorted([x for x in os.listdir(coingecko_data_path) if x.endswith(".json")])

    if len(files) == 0:
        return

    price_df = pd.DataFrame()
    if data_provider == "binance":
        for file in files:
            zip_file_path = os.path.join(binance_data_path, file)
            if not zip_file_path.endswith(".zip"):
                continue
            myzip = ZipFile(zip_file_path)
            with myzip.open(myzip.filelist[0]) as f:
                line = f.readline()
                header = 0 if line.decode("utf-8").startswith("open_time") else None
            df = pd.read_csv(myzip.open(myzip.filelist[0]), header=header).iloc[:, :11]
            df.columns = ["start_time", "open", "high", "low", "close", "volume", "end_time", "volume_usd", "n_trades", "taker_volume", "taker_volume_usd"]
            df.index = [pd.Timestamp(x + 1, unit="ms").to_datetime64() for x in df["end_time"]]
            df.index.name = "date"
            price_df = pd.concat([price_df, df])
        price_df.sort_index().to_csv(training_price_data_path)
    elif data_provider == "coingecko":
        for file in files:
            with open(os.path.join(coingecko_data_path, file), "r") as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                df.columns = ["timestamp", "open", "high", "low", "close"]
                df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.drop(columns=["timestamp"], inplace=True)
                df.set_index("date", inplace=True)
                price_df = pd.concat([price_df, df])
        price_df.sort_index().to_csv(training_price_data_path)

def load_frame(frame, timeframe):
    print(f"Loading data...")
    df = frame.loc[:, ['open', 'high', 'low', 'close']].dropna()
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].apply(pd.to_numeric)
    df['date'] = frame['date'].apply(pd.to_datetime)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    return df.resample(f'{timeframe}', label='right', closed='right', origin='end').mean()

def generate_features(df, token="ETHUSDT"):
    """Generate lag features for ETH and BTC as per xgboost_train_convert_ETH.py"""
    # Ensure we have both ETH and BTC data
    if token == "ETHUSDT":
        eth_df = df.copy()
        btc_files = download_data_binance("BTC", TRAINING_DAYS, REGION, "binance")
        format_data(btc_files, "binance")
        btc_df = pd.read_csv(training_price_data_path)
        btc_df = load_frame(btc_df, timeframe)
        btc_df.columns = [f"{col}_BTCUSDT" for col in btc_df.columns]
        df = eth_df.join(btc_df, how="inner")

    # Generate lag features (1 to 10) for OHLC
    for lag in range(1, 11):
        for col in ['open', 'high', 'low', 'close']:
            df[f'{col}_{token}_lag{lag}'] = df[col].shift(lag)
            if token == "ETHUSDT":
                df[f'{col}_BTCUSDT_lag{lag}'] = df[f'{col}_BTCUSDT'].shift(lag)

    # Add hour_of_day feature
    df['hour_of_day'] = df.index.hour

    # Drop rows with NaN values due to shifting
    df = df.dropna()
    return df

def train_model(timeframe):
    # Load the price data
    price_data = pd.read_csv(training_price_data_path)
    df = load_frame(price_data, timeframe)

    if MODEL == "XGBoost":
        # Generate features for XGBoost
        df = generate_features(df, token=TOKEN)
        print(df.tail())

        # Prepare features and target
        feature_cols = [f'f{i}' for i in range(81)]  # 81 features as per requirement
        X_train = df[[f'{col}_{TOKEN}_lag{lag}' for col in ['open', 'high', 'low', 'close'] for lag in range(1, 11)] +
                     [f'{col}_BTCUSDT_lag{lag}' for col in ['open', 'high', 'low', 'close'] for lag in range(1, 11)] +
                     ['hour_of_day']]
        X_train.columns = feature_cols  # Rename to f0, f1, ..., f80
        y_train = df['close'].shift(-1).dropna()
        X_train = X_train.iloc[:-1]  # Align with y_train

        print(f"Training data shape: {X_train.shape}, {y_train.shape}")

        # Train XGBoost model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': 0.05,
            'max_depth': 6
        }
        model = xgb.train(params, dtrain, num_boost_round=1000, early_stopping_rounds=10)

        # Save the model using pickle
        os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
        with open(model_file_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Trained XGBoost model saved to {model_file_path}")
    else:
        # Original model training logic
        print(df.tail())
        y_train = df['close'].shift(-1).dropna().values
        X_train = df[:-1]

        print(f"Training data shape: {X_train.shape}, {y_train.shape}")

        if MODEL == "LinearRegression":
            model = LinearRegression()
        elif MODEL == "SVR":
            model = SVR()
        elif MODEL == "KernelRidge":
            model = KernelRidge()
        elif MODEL == "BayesianRidge":
            model = BayesianRidge()
        else:
            raise ValueError("Unsupported model")
        
        model.fit(X_train, y_train)
        os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
        with open(model_file_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Trained model saved to {model_file_path}")

def get_inference(token, timeframe, region, data_provider):
    """Load model and predict current price."""
    with open(model_file_path, "rb") as f:
        loaded_model = pickle.load(f)

    # Get current price data
    if data_provider == "coingecko":
        current_df = download_coingecko_current_day_data(token, CG_API_KEY)
    else:
        current_df = download_binance_current_day_data(f"{token}USDT", region)
    X_new = load_frame(current_df, timeframe)
    X_new = generate_features(X_new, token=token)
    
    if MODEL == "XGBoost":
        # Prepare inference data for XGBoost
        feature_cols = [f'f{i}' for i in range(81)]
        X_new = X_new[[f'{col}_{token}_lag{lag}' for col in ['open', 'high', 'low', 'close'] for lag in range(1, 11)] +
                      [f'{col}_BTCUSDT_lag{lag}' for col in ['open', 'high', 'low', 'close'] for lag in range(1, 11)] +
                      ['hour_of_day']].iloc[-1:]
        X_new.columns = feature_cols
        dnew = xgb.DMatrix(X_new)

        print(X_new.tail())
        print(X_new.shape)

        current_price_pred = loaded_model.predict(dnew)
        return current_price_pred[0]
    else:
        # Original inference logic
        print(X_new.tail())
        print(X_new.shape)

        current_price_pred = loaded_model.predict(X_new)
        return current_price_pred[0]

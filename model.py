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
from config import data_base_path, model_file_path, TOKEN, MODEL, CG_API_KEY, TRAINING_DAYS, REGION, DATA_PROVIDER

binance_data_path = os.path.join(data_base_path, "binance")
coingecko_data_path = os.path.join(data_base_path, "coingecko")
training_price_data_path = os.path.join(data_base_path, "price_data.csv")

def download_data_binance(token, training_days, region):
    files = download_binance_daily_data(f"{token}USDT", training_days, region, binance_data_path)
    print(f"Downloaded {len(files)} new files for {token}USDT")
    return files

def download_data_coingecko(token, training_days):
    files = download_coingecko_data(token, training_days, coingecko_data_path, CG_API_KEY)
    print(f"Downloaded {len(files)} new files for {token}")
    return files

def download_data(token, training_days, region, data_provider):
    if data_provider == "coingecko":
        return download_data_coingecko(token, int(training_days))
    elif data_provider == "binance":
        return download_data_binance(token, training_days, region)
    else:
        raise ValueError("Unsupported data provider")

def format_data(files_btc, files_eth, data_provider):
    print(f"Formatting data with BTC files: {len(files_btc)}, ETH files: {len(files_eth)}")
    if not files_btc or not files_eth:
        print("No files provided for BTCUSDT or ETHUSDT")
        return
    
    if data_provider == "binance":
        files_btc = sorted([f for f in files_btc if "BTCUSDT" in os.path.basename(f) and f.endswith(".zip")])
        files_eth = sorted([f for f in files_eth if "ETHUSDT" in os.path.basename(f) and f.endswith(".zip")])
    elif data_provider == "coingecko":
        files_btc = sorted([x for x in files_btc if x.endswith(".json")])
        files_eth = sorted([x for x in files_eth if x.endswith(".json")])

    if len(files_btc) == 0 or len(files_eth) == 0:
        print("No valid files to process for BTCUSDT or ETHUSDT")
        return

    price_df_btc = pd.DataFrame()
    price_df_eth = pd.DataFrame()

    if data_provider == "binance":
        for file in files_btc:
            zip_file_path = os.path.join(binance_data_path, os.path.basename(file))
            if not os.path.exists(zip_file_path):
                print(f"File not found: {zip_file_path}")
                continue
            myzip = ZipFile(zip_file_path)
            with myzip.open(myzip.filelist[0]) as f:
                line = f.readline()
                header = 0 if line.decode("utf-8").startswith("open_time") else None
            df = pd.read_csv(myzip.open(myzip.filelist[0]), header=header).iloc[:, :11]
            df.columns = ["start_time", "open", "high", "low", "close", "volume", "end_time", "volume_usd", "n_trades", "taker_volume", "taker_volume_usd"]
            df['end_time'] = df['end_time'] // 1000  # 微秒 -> 毫秒
            df.index = [pd.Timestamp(x + 1, unit="ms").to_datetime64() for x in df["end_time"]]
            df.index.name = "date"
            price_df_btc = pd.concat([price_df_btc, df])

        for file in files_eth:
            zip_file_path = os.path.join(binance_data_path, os.path.basename(file))
            if not os.path.exists(zip_file_path):
                print(f"File not found: {zip_file_path}")
                continue
            myzip = ZipFile(zip_file_path)
            with myzip.open(myzip.filelist[0]) as f:
                line = f.readline()
                header = 0 if line.decode("utf-8").startswith("open_time") else None
            df = pd.read_csv(myzip.open(myzip.filelist[0]), header=header).iloc[:, :11]
            df.columns = ["start_time", "open", "high", "low", "close", "volume", "end_time", "volume_usd", "n_trades", "taker_volume", "taker_volume_usd"]
            df['end_time'] = df['end_time'] // 1000  # 微秒 -> 毫秒
            df.index = [pd.Timestamp(x + 1, unit="ms").to_datetime64() for x in df["end_time"]]
            df.index.name = "date"
            price_df_eth = pd.concat([price_df_eth, df])

    price_df_btc = price_df_btc.rename(columns=lambda x: f"{x}_BTCUSDT")
    price_df_eth = price_df_eth.rename(columns=lambda x: f"{x}_ETHUSDT")
    price_df = pd.concat([price_df_btc, price_df_eth], axis=1)

    # Generate features as in model_ref.py
    for pair in ["ETHUSDT", "BTCUSDT"]:
        for metric in ["open", "high", "low", "close"]:
            for lag in range(1, 11):
                price_df[f"{metric}_{pair}_lag{lag}"] = price_df[f"{metric}_{pair}"].shift(lag)

    price_df["hour_of_day"] = price_df.index.hour
    price_df = price_df.dropna()
    print(f"Formatted data shape: {price_df.shape}, columns: {price_df.columns.tolist()}")
    price_df.to_csv(training_price_data_path)
    print(f"Data saved to {training_price_data_path}")

def load_frame(frame, timeframe):
    print(f"Loading data with timeframe {timeframe}...")
    df = frame.loc[:, ['open', 'high', 'low', 'close']].dropna()
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].apply(pd.to_numeric)
    
    df['date'] = frame['date']
    try:
        df['date'] = pd.to_numeric(df['date'])
        if df['date'].max() > 4102444800000:  # 超过 2100-01-01 的毫秒时间戳
            df['date'] = df['date'] // 1000
        df['date'] = pd.to_datetime(df['date'], unit='ms', errors='coerce')
    except ValueError:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    df = df.dropna(subset=['date'])
    if df.empty:
        raise ValueError("No valid dates found in the data after cleaning.")
    
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    return df.resample(f'{timeframe}', label='right', closed='right', origin='end').mean()

def generate_features(df, token="ETHUSDT", data_provider=DATA_PROVIDER):
    print(f"Generating features for token: {token}, data_provider: {data_provider}")
    eth_df = df.copy()
    print(f"ETH data shape before processing: {eth_df.shape}")
    print(f"ETH data columns: {eth_df.columns.tolist()}")

    if token == "ETHUSDT":
        print("Downloading BTC data...")
        btc_files = download_data("BTC", TRAINING_DAYS, REGION, data_provider)
        format_data(btc_files, [], data_provider)  # Only BTC data
        if not os.path.exists(training_price_data_path):
            raise FileNotFoundError(f"Training data file not found at {training_price_data_path}")
        combined_df = pd.read_csv(training_price_data_path, index_col='date', parse_dates=True)
        df = combined_df
        print(f"Combined data loaded: {df.shape}, columns: {df.columns.tolist()}")

    for lag in range(1, 11):
        for col in ['open', 'high', 'low', 'close']:
            col_name = f'{col}_{token}_lag{lag}'
            df[col_name] = df[col].shift(lag)
            print(f"Generated {col_name}")
            if token == "ETHUSDT":
                btc_col = f'{col}_BTCUSDT'
                btc_lag_col = f'{col}_BTCUSDT_lag{lag}'
                if btc_col not in df.columns:
                    raise ValueError(f"Column {btc_col} not found in DataFrame after join")
                df[btc_lag_col] = df[btc_col].shift(lag)
                print(f"Generated {btc_lag_col}")

    df['hour_of_day'] = df.index.hour
    df = df.dropna()
    print(f"Final features generated: {df.columns.tolist()}")
    print(f"Final data shape after dropna: {df.shape}")
    return df

def train_model(timeframe):
    print(f"Starting train_model with timeframe: {timeframe}")
    # Download and format data for both BTC and ETH
    files_btc = download_data("BTC", TRAINING_DAYS, REGION, DATA_PROVIDER)
    files_eth = download_data("ETH", TRAINING_DAYS, REGION, DATA_PROVIDER)
    format_data(files_btc, files_eth, DATA_PROVIDER)

    # Load combined data
    price_data = pd.read_csv(training_price_data_path, index_col='date', parse_dates=True)
    df = load_frame(price_data, timeframe)
    print(f"Loaded data: {df.shape}, columns: {df.columns.tolist()}")

    if MODEL == "XGBoost":
        df = generate_features(df, token=TOKEN, data_provider=DATA_PROVIDER)
        print("DataFrame after generate_features:")
        print(df.tail())

        feature_cols = [f'f{i}' for i in range(81)]
        required_cols = [f'{col}_{TOKEN}_lag{lag}' for col in ['open', 'high', 'low', 'close'] for lag in range(1, 11)] + \
                        [f'{col}_BTCUSDT_lag{lag}' for col in ['open', 'high', 'low', 'close'] for lag in range(1, 11)] + \
                        ['hour_of_day']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
        X_train = df[required_cols]
        X_train.columns = feature_cols
        y_train = df['close'].shift(-1).dropna()
        X_train = X_train.iloc[:-1]

        print(f"Training data shape: {X_train.shape}, {y_train.shape}")

        dtrain = xgb.DMatrix(X_train, label=y_train)
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': 0.05,
            'max_depth': 6
        }
        model = xgb.train(params, dtrain, num_boost_round=1000, early_stopping_rounds=10)

        os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
        with open(model_file_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Trained XGBoost model saved to {model_file_path}")
    else:
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
    with open(model_file_path, "rb") as f:
        loaded_model = pickle.load(f)

    if data_provider == "coingecko":
        current_df = download_coingecko_current_day_data(token, CG_API_KEY)
    else:
        current_df = download_binance_current_day_data(f"{token}USDT", region)
    X_new = load_frame(current_df, timeframe)
    X_new = generate_features(X_new, token=token, data_provider=data_provider)
    
    if MODEL == "XGBoost":
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
        print(X_new.tail())
        print(X_new.shape)
        current_price_pred = loaded_model.predict(X_new)
        return current_price_pred[0]

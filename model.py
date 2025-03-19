import json
import os
import pickle
from zipfile import ZipFile
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
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

    # Generate features
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
    
    if 'end_time' in frame.columns:
        df.index = pd.to_datetime(frame['end_time'] + 1, unit='ms', errors='coerce')
    else:
        df.index = pd.to_datetime(df.index, errors='coerce')
    
    df = df.dropna(subset=['open'])
    if df.empty:
        raise ValueError("No valid data found after cleaning.")
    
    df.sort_index(inplace=True)
    # Ensure index is DatetimeIndex after resampling
    df = df.resample(f'{timeframe}', label='right', closed='right', origin='end').mean()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index is not a DatetimeIndex after resampling.")
    print(f"Loaded frame shape: {df.shape}, index type: {type(df.index)}")
    return df

def generate_features(df, token="ETHUSDT", data_provider=DATA_PROVIDER, timeframe='1h'):
    print(f"Generating features for token: {token}, data_provider: {data_provider}")
    print(f"Data shape before processing: {df.shape}")
    print(f"Data columns: {df.columns.tolist()}")

    # Load and resample historical data to match timeframe
    if os.path.exists(training_price_data_path):
        hist_df = pd.read_csv(training_price_data_path, index_col='date', parse_dates=True)
        hist_df = hist_df.tail(14400)  # Last 10 days for efficiency
        hist_df_eth = hist_df[[f'{col}_{token}USDT' for col in ['open', 'high', 'low', 'close']]].resample(timeframe).mean()
        hist_df_btc = hist_df[[f'{col}_BTCUSDT' for col in ['open', 'high', 'low', 'close']]].resample(timeframe).mean()
        
        # Combine ETH and BTC data with real-time data
        combined_df = pd.concat([hist_df_eth, df[[f'{col}_{token}USDT' for col in ['open', 'high', 'low', 'close']]]], axis=0, join='outer')
        combined_df = pd.concat([combined_df, hist_df_btc, df[[f'{col}_BTCUSDT' for col in ['open', 'high', 'low', 'close']]]], axis=1, join='outer')
        
        # Reset index to ensure DatetimeIndex after merging
        combined_df.index = pd.to_datetime(combined_df.index, errors='coerce')
        print(f"After merging ETH and BTC, index type: {type(combined_df.index)}, shape: {combined_df.shape}")
        
        # Handle duplicate columns by keeping the first occurrence
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
        print(f"After removing duplicates, shape: {combined_df.shape}, columns: {combined_df.columns.tolist()}")
        df = combined_df
    else:
        print("No historical data found, using only real-time data.")

    # Generate ETH lag features
    for metric in ["open", "high", "low", "close"]:
        for lag in range(1, 11):
            df[f"{metric}_{token}USDT_lag{lag}"] = df[f"{metric}_{token}USDT"].shift(lag)

    # Generate BTC lag features
    for metric in ["open", "high", "low", "close"]:
        for lag in range(1, 11):
            df[f"{metric}_BTCUSDT_lag{lag}"] = df[f"{metric}_BTCUSDT"].shift(lag)

    # Ensure index is DatetimeIndex before accessing hour
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Failed to convert index to DatetimeIndex.")
    
    df['hour_of_day'] = df.index.hour
    df = df.dropna()
    if df.empty:
        raise ValueError("Generated DataFrame is empty after dropping NaN values.")
    
    print(f"Features generated: {df.columns.tolist()}")
    print(f"Final data shape: {df.shape}")
    return df

def train_model(timeframe):
    print(f"Starting train_model with timeframe: {timeframe}")
    if not os.path.exists(training_price_data_path):
        raise FileNotFoundError(f"Training data file not found at {training_price_data_path}. Run update_data first.")
    
    price_data = pd.read_csv(training_price_data_path, index_col='date', parse_dates=True)
    df = price_data
    print(f"Loaded data: {df.shape}, columns: {df.columns.tolist()}")

    if MODEL == "XGBoost":
        print("DataFrame before training:")
        print(df.tail())

        feature_cols = [f'f{i}' for i in range(81)]
        required_cols = [f'{col}_{TOKEN}USDT_lag{lag}' for col in ['open', 'high', 'low', 'close'] for lag in range(1, 11)] + \
                        [f'{col}_BTCUSDT_lag{lag}' for col in ['open', 'high', 'low', 'close'] for lag in range(1, 11)] + \
                        ['hour_of_day']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
        X = df[required_cols]
        X.columns = feature_cols
        y = df[f'close_{TOKEN}USDT'].shift(-1).dropna()
        X = X.iloc[:-1]

        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

        print(f"Training data shape: {X_train.shape}, {y_train.shape}")
        print(f"Validation data shape: {X_val.shape}, {y_val.shape}")

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': 0.05,
            'max_depth': 6
        }
        model = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dval, 'validation')], early_stopping_rounds=10)

        # Calculate and print performance metrics
        y_train_pred = model.predict(dtrain)
        y_val_pred = model.predict(dval)

        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)

        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_r2 = r2_score(y_val, y_val_pred)

        print(f"Training MAE: {train_mae:.4f}")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Validation MAE: {val_mae:.4f}")
        print(f"Validation RMSE: {val_rmse:.4f}")
        print(f"Validation R²: {val_r2:.4f}")

        os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
        with open(model_file_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Trained XGBoost model saved to {model_file_path}")
    else:
        print(df.tail())
        y_train = df[f'close_{TOKEN}USDT'].shift(-1).dropna().values
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
        
        # Calculate and print performance metrics for non-XGBoost models
        y_train_pred = model.predict(X_train)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Training R²: {train_r2:.4f}")

        os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
        with open(model_file_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Trained model saved to {model_file_path}")

def get_inference(token, timeframe, region, data_provider):
    with open(model_file_path, "rb") as f:
        loaded_model = pickle.load(f)

    if data_provider == "coingecko":
        current_df_eth = download_coingecko_current_day_data(token, CG_API_KEY)
        current_df_btc = download_coingecko_current_day_data("bitcoin", CG_API_KEY)
    else:
        current_df_eth = download_binance_current_day_data(f"{token}USDT", region)
        current_df_btc = download_binance_current_day_data("BTCUSDT", region)
    
    print(f"Raw ETH data shape: {current_df_eth.shape}, columns: {current_df_eth.columns.tolist()}")
    print(f"Raw BTC data shape: {current_df_btc.shape}, columns: {current_df_btc.columns.tolist()}")

    X_new_eth = load_frame(current_df_eth, timeframe)
    X_new_btc = load_frame(current_df_btc, timeframe)
    
    X_new_eth = X_new_eth.rename(columns={
        'open': f'open_{token}USDT',
        'high': f'high_{token}USDT',
        'low': f'low_{token}USDT',
        'close': f'close_{token}USDT'
    })
    X_new_btc = X_new_btc.rename(columns={
        'open': 'open_BTCUSDT',
        'high': 'high_BTCUSDT',
        'low': 'low_BTCUSDT',
        'close': 'close_BTCUSDT'
    })

    X_new = pd.concat([X_new_eth, X_new_btc], axis=1)
    X_new = generate_features(X_new, token=token, data_provider=data_provider, timeframe=timeframe)
    
    if MODEL == "XGBoost":
        feature_cols = [f'f{i}' for i in range(81)]
        X_new = X_new[[f'{col}_{token}USDT_lag{lag}' for col in ['open', 'high', 'low', 'close'] for lag in range(1, 11)] +
                      [f'{col}_BTCUSDT_lag{lag}' for col in ['open', 'high', 'low', 'close'] for lag in range(1, 11)] +
                      ['hour_of_day']].iloc[-1:]
        if X_new.empty:
            raise ValueError("No valid data available for inference after feature generation.")
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

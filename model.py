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
eth_price_data_path = os.path.join(data_base_path, "eth_price_data.csv")
btc_price_data_path = os.path.join(data_base_path, "btc_price_data.csv")

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

def format_data(files, data_provider, output_path):
    if not files:
        print("No new files to process")
        return
    
    print(f"Files received: {files}")
    
    token_prefix = os.path.basename(files[0]).split('-')[0] if files else "UNKNOWN"
    if data_provider == "binance":
        files = [f for f in files if os.path.basename(f).startswith(token_prefix)]
    elif data_provider == "coingecko":
        files = [f for f in files if os.path.basename(f).endswith(".json")]

    if len(files) == 0:
        print("No matching files found after filtering")
        return

    price_df = pd.DataFrame()
    if data_provider == "binance":
        for file in files:
            zip_file_path = os.path.join(binance_data_path, file) if not os.path.isabs(file) else file
            if not zip_file_path.endswith(".zip"):
                continue
            myzip = ZipFile(zip_file_path)
            with myzip.open(myzip.filelist[0]) as f:
                line = f.readline()
                header = 0 if line.decode("utf-8").startswith("open_time") else None
            df = pd.read_csv(myzip.open(myzip.filelist[0]), header=header).iloc[:, :11]
            df.columns = ["start_time", "open", "high", "low", "close", "volume", "end_time", "volume_usd", "n_trades", "taker_volume", "taker_volume_usd"]
            print(f"Processing {file}, end_time sample: {df['end_time'].head(5)}")
            df['end_time'] = df['end_time'] // 1000  # 微秒 -> 毫秒
            df.index = [pd.Timestamp(x + 1, unit="ms").to_datetime64() for x in df["end_time"]]
            df.index.name = "date"
            price_df = pd.concat([price_df, df])
        if not price_df.empty:
            price_df.sort_index().to_csv(output_path)
            print(f"Data saved to {output_path}")
        else:
            print("No data processed for Binance")
    elif data_provider == "coingecko":
        for file in files:
            file_path = os.path.join(coingecko_data_path, file) if not os.path.isabs(file) else file
            with open(file_path, "r") as f:
                data 

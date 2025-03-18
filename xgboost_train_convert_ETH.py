import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import onnxmltools
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

# Load the data
data_path = './data/ETH/ETHUSDT_1h_spot_forecast_training.csv'
data = pd.read_csv(data_path)

# Preprocess the data
# Assuming the target variable is 'target_BTCUSDT' and features are all other columns
X = data.drop(columns=['target_ETHUSDT', 'timestamp'])
y = data['target_ETHUSDT']

# Rename feature columns to match XGBoost's expected pattern
X.columns = [f'f{i}' for i in range(X.shape[1])]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'eta': 0.05,
    'max_depth': 6
}

model = xgb.train(params, dtrain, evals=[(dval, 'validation')], num_boost_round=1000, early_stopping_rounds=10)

# Evaluate the model
y_pred = model.predict(dval)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f'Validation RMSE: {rmse}')

# Save the model
model.save_model('./models/xgboost_model_eth.json')

# Convert the model to ONNX format
initial_type = [('candles', FloatTensorType([1, X_train.shape[1]]))]
onnx_model = convert_xgboost(model, initial_types=initial_type)
onnxmltools.utils.save_model(onnx_model, './models/xgboost_model_eth.onnx')
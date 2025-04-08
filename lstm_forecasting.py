# from google.colab import drive
# drive.mount('/content/drive')

### Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json

### Hyperparameters
WINDOW_SIZE = 24
EPOCHS = 100
LEARNING_RATE = 0.001
TYPE = 'Univariate' # 'Univariate' or 'Multivariate'
ARCHITECTURE = 'LSTM' # 'LSTM' or 'BiLSTM'
MODEL = f'{TYPE}_{ARCHITECTURE}'
DATASET = 'AMBUJACEM' #'AMBUJACEM' or 'ICICI' or 'WIPRO' or 'NTPC'
ROOT_DIR = '/content/drive/MyDrive/BTechProject'
NUM_POINTS = 1600

## Preprocess data
def preprocess_data(df, date_col='date', drop_cols=['Unnamed: 0', 'date'], datetime_format='%Y-%m-%d %H:%M:%S+05:30'):
  """
  Preprocesses the input DataFrame by performing the following steps:
  1. Drops rows with NaN values.
  2. Converts the specified date column to datetime and sets it as the index.
  3. Drops specified columns from the DataFrame.
  4. Normalizes the data using MinMaxScaler.
  5. Converts the normalized data back to a DataFrame with the original index.

  Parameters:
  df (pd.DataFrame): The input DataFrame to preprocess.
  date_col (str): The name of the column containing date information. Default is 'date'.
  drop_cols (list): A list of columns to drop from the DataFrame. Default is ['Unnamed: 0', 'date'].
  datetime_format (str): The format of the datetime string in the date column. Default is '%Y-%m-%d %H:%M:%S+05:30'.

  Returns:
  pd.DataFrame: The preprocessed and normalized DataFrame.
  """
  # Drop NaN values
  df.dropna(inplace=True)

  # Convert the index to datetime
  df.index = pd.to_datetime(df[date_col], format=datetime_format)

  # Drop specified columns
  df_cleaned = df.drop(drop_cols, axis=1)

  # Normalize the data
  scaler = MinMaxScaler()
  df_normalized_values = scaler.fit_transform(df_cleaned.values)

  # Convert back to DataFrame
  df_normalized = pd.DataFrame(df_normalized_values, columns=df_cleaned.columns)
  df_normalized.index = df.index

  return df_normalized

def load_and_preprocess_data(dataset, root_dir):
  """
  Load and preprocess data from a CSV file.

  This function reads a CSV file, preprocesses the data, calculates and prints correlations,
  and returns several DataFrame objects and the number of variables.

  Args:
    dataset (str): The name of the dataset (without file extension).
    root_dir (str): The root directory where the CSV file is located.

  Returns:
    tuple: A tuple containing the following elements:
      - df4 (pd.DataFrame): The preprocessed DataFrame.
      - close (pd.Series): The 'close' column from the preprocessed DataFrame.
      - df5 (pd.DataFrame): A DataFrame containing selected columns for further analysis.
      - num_of_variables (int): The number of variables in df5.
      - df6 (pd.DataFrame): A DataFrame containing only the 'open', 'high', 'low', and 'close' columns.
  """
  df = pd.read_csv(f"{root_dir}/{dataset}_edited_all_columns.csv")
  df.head()
  df4 = preprocess_data(df)

  # Plot the 'close' column
  #df4.iloc[::100].plot(kind='line', fontsize=12, figsize=(16, 8))

  # Call the function
  calculate_and_print_correlations(df4, target_column='close')

  close = df4['close']
  #close.plot()

  df5 = df4[['open', 'high', 'low', 'close',
          'ema5', 'sma5', 'TRIMA5', 'lowerband', 'middleband',
          'upperband', 'KAMA10', 'volume']]

  num_of_variables = df5.columns.size

  df6 = df4[['open', 'high', 'low', 'close']]

  return df4, close, df5, num_of_variables, df6

### Utils
def calculate_and_print_correlations(df, target_column='close'):
  """
  Calculate and print the correlation of all columns with the target column.

  Parameters:
  df (DataFrame): The input DataFrame.
  target_column (str): The column to calculate correlations with.
  """
  # Calculate the correlation of all columns with the target column
  correlation_with_target = df.corr()[target_column]

  # Drop the correlation of the target column with itself
  correlation_with_target = correlation_with_target.drop(target_column)

  # Sort the correlation values in descending order
  sorted_correlation_with_target = correlation_with_target.sort_values(ascending=False)

  # Print the sorted correlations
  print(sorted_correlation_with_target)

def train_model(model, X_train, y_train, X_val, y_val,
        learning_rate=0.001, epochs=10, patience=5, 
        model_save_path='model.h5'):
  """
  Train the model with the given training and validation data.

  Parameters:
  model (Sequential): The model to be trained.
  X_train (ndarray): Training data features.
  y_train (ndarray): Training data labels.
  X_val (ndarray): Validation data features.
  y_val (ndarray): Validation data labels.
  learning_rate (float): Learning rate for the optimizer. Default is 0.001.
  epochs (int): Number of epochs to train the model. Default is 10.
  patience (int): Number of epochs with no improvement after which training will be stopped. Default is 5.
  model_save_path (str): Path to save the trained model. Default is 'model.h5'.

  Returns:
  history (History): A record of training loss values and metrics values at successive epochs.
  """
  # Compile the model
  model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=learning_rate),
        metrics=[RootMeanSquaredError()])

  # Train the model
  history = model.fit(X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=[
            tf.keras.callbacks.EarlyStopping(
              monitor='val_loss',
              patience=patience,
              restore_best_weights=True)
            ])
  
  # Save the trained model
  model.save(model_save_path)
  print(f'Model saved to {model_save_path}')

  return history

def save_history_as_json(history, json_save_path='history.json'):
  # Convert the history.history dict to JSON
  history_dict = history.history
  with open(json_save_path, 'w') as f:
    json.dump(history_dict, f)
  print(f'History saved to {json_save_path}')

def evaluate_model(model, X_test, y_test, num_points=1600):
  """
  Evaluate the model on the test set and calculate performance metrics.

  Parameters:
  model (Sequential): The trained model.
  X_test (ndarray): The test set features.
  y_test (ndarray): The test set labels.
  num_points (int): The number of points to plot.

  Returns:
  test_results (DataFrame): DataFrame containing test predictions and actual values.
  metrics (dict): Dictionary containing performance metrics.
  """
  # Generate predictions on the test set
  test_predictions = model.predict(X_test).flatten()

  # Create a DataFrame with predictions and actual values
  test_results = pd.DataFrame(data={'Test Predictions': test_predictions, 'Actuals': y_test})

  # Calculate performance metrics
  r2 = r2_score(y_test, test_predictions)
  mse = mean_squared_error(y_test, test_predictions)
  mae = mean_absolute_error(y_test, test_predictions)
  rmse = np.sqrt(mse)
  mape = np.mean(np.abs((y_test - test_predictions) / y_test)) * 100

  # Store the performance metrics in a dictionary
  metrics = {'r2': r2, 'mse': mse, 'mae': mae, 'rmse': rmse, 'mape': mape}

  return test_results, metrics

def save_results_to_files(test_results, metrics, results_filename='test_results.csv', metrics_filename='model_metrics.json'):
  """
  Save test results to a CSV file and metrics to a JSON file.
  
  Parameters:
  test_results (DataFrame): DataFrame containing test predictions and actual values.
  metrics (dict): Dictionary containing performance metrics.
  results_filename (str): Filename for the CSV containing test results.
  metrics_filename (str): Filename for the JSON containing metrics.
  """
  # Save test results to CSV
  test_results.to_csv(results_filename, index=False)
  
  # Save metrics to JSON
  with open(metrics_filename, 'w') as f:
    json.dump(metrics, f, indent=4)
  
  print(f"Test results saved to {results_filename}")
  print(f"Metrics saved to {metrics_filename}")

def plot_predictions(test_results, num_points=1600):
  """
  Plot the test predictions and actual values.

  Parameters:
  test_results (DataFrame): DataFrame containing test predictions and actual values.
  num_points (int): The number of points to plot.
  """
  plt.figure(figsize=(10, 6))
  plt.plot(test_results['Test Predictions'][:num_points], label='Test Predictions')
  plt.plot(test_results['Actuals'][:num_points], label='Actuals')

  # Add legend and show plot
  plt.legend()
  plt.title('Test Predictions vs Actuals')
  plt.xlabel('Data Points')
  plt.ylabel('Values')
  plt.show()


## Univariate models
"""
- Using `close` predict `close`

#### Convert the df to a matrix
- So, what we are going to do is - use the data at 1pm, 2pm, 3pm, 4pm, 5pm to get prediction for 6pm
- Then, we will use 2,3,4,5,6 to get 7pm prediction and so on
- So, we have a sliding window of 5 horus to predict the temp in the next hour

##### This is exactly given in the matrix below
- So, the 1,2,3,4,5 pm data points will be `X` and the prediction, 6 will be `y`

```python
[[[1], [2], [3], [4], [5]]] [6]
[[[2], [3], [4], [5], [6]]] [7]
[[[3], [4], [5], [6], [7]]] [8]
```
"""

## UNIVARIATE - MAKE WINDOW (X) AND OUTPUT (y)
## This function converts the dataframe to the matrix format as given above
def df_to_X_y_Univariate(df, window_size=24):
  """
  Convert the DataFrame to input-output pairs for univariate time series prediction.

  Parameters:
  df (pd.DataFrame): The input DataFrame containing the time series data.
  window_size (int): The size of the window for the input sequence. Default is 24.

  Returns:
  tuple: A tuple containing the following elements:
    - X (ndarray): The input sequences.
    - y (ndarray): The corresponding output values.
  """
  # Convert the DataFrame to a numpy array
  df_as_np = df.to_numpy()
  
  # Initialize empty lists to store input sequences (X) and output values (y)
  X = []
  y = []
  
  # Loop through the data to create input-output pairs
  for i in range(len(df_as_np) - window_size):
    # Create a window of data for the input sequence
    row = [[a] for a in df_as_np[i:i + window_size]]
    X.append(row)
    
    # The output value is the data point immediately after the window
    label = df_as_np[i + window_size]
    y.append(label)
  
  # Convert the lists to numpy arrays and return
  return np.array(X), np.array(y)

### Test train validation split
def prepare_univariate_data(df, window_size=24, train_ratio=0.7, val_ratio=0.15):
  """
  Prepares univariate time series data for training, validation, and testing.

  Parameters:
  df (pd.DataFrame): The input dataframe containing the time series data.
  window_size (int): The size of the window to create the input sequences. Default is 24.
  train_ratio (float): The ratio of the data to be used for training. Default is 0.7.
  val_ratio (float): The ratio of the data to be used for validation. Default is 0.15.

  Returns:
  tuple: A tuple containing three tuples:
    - (X_train, y_train): Training data and labels.
    - (X_val, y_val): Validation data and labels.
    - (X_test, y_test): Testing data and labels.
  """
  X, y = df_to_X_y_Univariate(df, window_size)
  TOTAL_SIZE = len(X)
  train_size = int(train_ratio * TOTAL_SIZE)
  val_size = int(val_ratio * TOTAL_SIZE)

  train_limit = train_size
  val_limit = train_size + val_size

  X_train, y_train = X[:train_limit], y[:train_limit]
  X_val, y_val = X[train_limit:val_limit], y[train_limit:val_limit]
  X_test, y_test = X[val_limit:], y[val_limit:]

  return (X_train, y_train), (X_val, y_val), (X_test, y_test)

### Univariate LSTM model
class UnivariateLSTMModel:
  def __init__(self, window_size):
    """
    Initialize the Univariate LSTM Model.

    Parameters:
    window_size (int): The size of the window for the input sequence.
    """
    self.window_size = window_size
    self.model = self.build_model()

  def build_model(self):
    """
    Build the LSTM model.

    Returns:
    model (Sequential): The compiled LSTM model.
    """
    model = Sequential()
    model.add(InputLayer((self.window_size, 1)))
    model.add(LSTM(64))
    model.add(Dense(8, 'relu'))
    model.add(Dense(1, 'linear'))
    return model

  def summary(self):
    """
    Print the summary of the model.
    """
    self.model.summary()

### Univariate Bidirectional LSTM model
class UnivariateBiLSTMModel:
  def __init__(self, window_size):
    """
    Initialize the Univariate Bidirectional LSTM Model.

    Parameters:
    window_size (int): The size of the window for the input sequence.
    """
    self.window_size = window_size
    self.model = self.build_model()

  def build_model(self):
    """
    Build the Bidirectional LSTM model.

    Returns:
    model (Sequential): The compiled Bidirectional LSTM model.
    """
    model = Sequential()
    model.add(InputLayer((self.window_size, 1)))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(8, 'relu'))
    model.add(Dense(1, 'linear'))
    return model

  def summary(self):
    """
    Print the summary of the model.
    """
    self.model.summary()

## Multivariate models
"""
- Using `open`, `high`, `low`, `close`, `volume` to predict `close`

#### Convert the df to a matrix
- So, what we are going to do is - use the data at 1pm, 2pm, 3pm, 4pm, 5pm to get prediction for 6pm
- Then, we will use 2,3,4,5,6 to get 7pm prediction and so on
- So, we have a sliding window of 5 hours to predict the `close` in the next hour

##### This is exactly given in the matrix below
- So, the 1,2,3,4,5 pm data points will be `X` and the prediction, 6 will be `y`
"""

## MULTIVARIATE - MAKE WINDOW (X) AND OUTPUT (y)
## This function converts the dataframe to the matrix format as given above
def df_to_X_y_Multivariate(df, window_size=24):
  """
  Convert the DataFrame to input-output pairs for multivariate time series prediction.

  Parameters:
  df (pd.DataFrame): The input DataFrame containing the time series data.
  window_size (int): The size of the window for the input sequence. Default is 24.

  Returns:
  tuple: A tuple containing the following elements:
    - X (ndarray): The input sequences.
    - y (ndarray): The corresponding output values.
  """
  # Convert the DataFrame to a numpy array
  df_as_np = df.to_numpy()
  
  # Initialize empty lists to store input sequences (X) and output values (y)
  X = []
  y = []
  
  # Loop through the data to create input-output pairs
  for i in range(len(df_as_np) - window_size):
    # Create a window of data for the input sequence
    row = [r for r in df_as_np[i:i + window_size]]
    X.append(row)
    
    # The output value is the first data point immediately after the window
    label = df_as_np[i + window_size][0]
    y.append(label)
  
  # Convert the lists to numpy arrays and return
  return np.array(X), np.array(y)

def prepare_multivariate_data(df, window_size=24, train_ratio=0.7, val_ratio=0.15):
  """
  Prepares multivariate time series data for training, validation, and testing.

  Parameters:
  df (pd.DataFrame): The input dataframe containing the time series data.
  window_size (int): The size of the window to create the sequences. Default is 24.
  train_ratio (float): The ratio of the data to be used for training. Default is 0.7.
  val_ratio (float): The ratio of the data to be used for validation. Default is 0.15.

  Returns:
  tuple: A tuple containing three tuples:
    - (X_train, y_train): Training data and labels.
    - (X_val, y_val): Validation data and labels.
    - (X_test, y_test): Testing data and labels.
  """
  X, y = df_to_X_y_Multivariate(df, window_size)
  print(X.shape, y.shape)

  TOTAL_SIZE = len(X)
  train_size = int(train_ratio * TOTAL_SIZE)
  val_size = int(val_ratio * TOTAL_SIZE)

  train_limit = train_size
  val_limit = train_size + val_size

  X_train, y_train = X[:train_limit], y[:train_limit]
  X_val, y_val = X[train_size:val_limit], y[train_size:val_limit]
  X_test, y_test = X[val_limit:], y[val_limit:]

  print(X_train.shape, y_train.shape)
  print(X_val.shape, y_val.shape)
  print(X_test.shape, y_test.shape)

  return (X_train, y_train), (X_val, y_val), (X_test, y_test)

## Multivariate Single LSTM

class MultivariateLSTMModel:
  def __init__(self, window_size, num_of_variables):
    """
    Initialize the Multivariate LSTM Model.

    Parameters:
    window_size (int): The size of the window for the input sequence.
    num_of_variables (int): The number of variables in the input sequence.
    """
    self.window_size = window_size
    self.num_of_variables = num_of_variables
    self.model = self.build_model()

  def build_model(self):
    """
    Build the LSTM model.

    Returns:
    model (Sequential): The compiled LSTM model.
    """
    model = Sequential()
    model.add(InputLayer((self.window_size, self.num_of_variables)))
    model.add(LSTM(64))
    model.add(Dense(8, 'relu'))
    model.add(Dense(1, 'linear'))
    return model

  def summary(self):
    """
    Print the summary of the model.
    """
    self.model.summary()

## Multivariate Bidirectional LSTM

class MultivariateBiLSTMModel:
  def __init__(self, window_size, num_of_variables):
    """
    Initialize the Multivariate Bidirectional LSTM Model.

    Parameters:
    window_size (int): The size of the window for the input sequence.
    num_of_variables (int): The number of variables in the input sequence.
    """
    self.window_size = window_size
    self.num_of_variables = num_of_variables
    self.model = self.build_model()

  def build_model(self):
    """
    Build the Bidirectional LSTM model.

    Returns:
    model (Sequential): The compiled Bidirectional LSTM model.
    """
    model = Sequential()
    model.add(InputLayer((self.window_size, self.num_of_variables)))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(8, 'relu'))
    model.add(Dense(1, 'linear'))
    return model

  def summary(self):
    """
    Print the summary of the model.
    """
    self.model.summary()

def main():
  # Dataloader
  df4, close, df5, NUM_OF_VARIABLES, df6 = load_and_preprocess_data(DATASET, ROOT_DIR)

  if TYPE == 'Univariate':
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_univariate_data(close, WINDOW_SIZE)
    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)
  elif TYPE == 'Multivariate':
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_multivariate_data(df5, WINDOW_SIZE)
    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)
  
  # Instantiate and summarize the model
  if MODEL == 'Univariate_LSTM':
    model = UnivariateLSTMModel(WINDOW_SIZE)
  elif MODEL == 'Univariate_BiLSTM':
    model = UnivariateBiLSTMModel(WINDOW_SIZE)
  elif MODEL == 'Multivariate_LSTM':
    model = MultivariateLSTMModel(WINDOW_SIZE, NUM_OF_VARIABLES)
  elif MODEL == 'Multivariate_BiLSTM':
    model = MultivariateBiLSTMModel(WINDOW_SIZE, NUM_OF_VARIABLES)

  print(model.summary())  
  name = f'{ROOT_DIR}/{MODEL}_{DATASET}'
  model_save_path = f'{name}.h5'
  history = train_model(model, X_train, y_train, X_val, y_val,
                        learning_rate=LEARNING_RATE, epochs=EPOCHS,
                        model_save_path=model_save_path)
  
  save_history_as_json(history, json_save_path=f'{name}_history.json')

  loaded_model = tf.keras.models.load_model(model_save_path)
  test_results, metrics = evaluate_model(loaded_model, X_test, y_test, num_points=NUM_POINTS)
  save_results_to_files(test_results, metrics, 
                        results_filename=f'{name}_test_results.csv', 
                        metrics_filename=f'{name}_model_metrics.json')
  plot_predictions(test_results, num_points=NUM_POINTS)

if __name__ == '__main__':
  main()

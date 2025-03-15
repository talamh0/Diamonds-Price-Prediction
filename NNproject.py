import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

# Load the Diamonds dataset
DiamondDS = sns.load_dataset("diamonds")

# Separate features and target
X = DiamondDS.drop(columns=['price'])
y = DiamondDS['price']

# Encode categorical features using one-hot encoding
X_encoded = pd.get_dummies(X, drop_first=True)

# Normalize feature values using standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_encoded.columns)

# Split the data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Build a two-layer feedforward neural network
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(10, activation='relu'))  # First hidden layer with 10 neurons
model.add(Dense(5, activation='relu'))   # Second hidden layer with 5 neurons
model.add(Dense(1))  # Output layer with a single node

# Display model architecture
model.summary()

# Compile and train the model with a learning rate of 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
history_001 = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=0)

# Plot the loss curve for lr=0.001
plt.plot(history_001.history['loss'], label='Training Loss (lr=0.001)')
plt.plot(history_001.history['val_loss'], label='Validation Loss (lr=0.001)')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Loss Curve (lr=0.001)')
plt.show()

# Retrain the model with a learning rate of 0.3
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.3), loss='mean_squared_error')
history_03 = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=0)

# Plot the loss curve for lr=0.3
plt.plot(history_03.history['loss'], label='Training Loss (lr=0.3)')
plt.plot(history_03.history['val_loss'], label='Validation Loss (lr=0.3)')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Loss Curve (lr=0.3)')
plt.show()

# Evaluate the model on the test set
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R2 Score: {r2}')

# Scatter plot comparing true labels and predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')
plt.show()

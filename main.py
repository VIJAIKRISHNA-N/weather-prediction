import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import timedelta

# Load data
df = pd.read_csv('weather.csv', parse_dates=['date'])
df = df.sort_values('date')

# Convert date to ordinal for regression
df['date_ordinal'] = df['date'].map(pd.Timestamp.toordinal)

# Train/Test Split
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Model
model = LinearRegression()
model.fit(train[['date_ordinal']], train['temperature'])

# Predict on test set
test['predicted_temp'] = model.predict(test[['date_ordinal']])

# Evaluate
mse = mean_squared_error(test['temperature'], test['predicted_temp'])
print(f"Mean Squared Error: {mse:.2f}")

# Plot
plt.figure(figsize=(10,5))
plt.plot(df['date'], df['temperature'], label='Actual', color='blue')
plt.plot(test['date'], test['predicted_temp'], label='Predicted', color='red')
plt.title('Temperature Prediction')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.tight_layout()
plt.show()

# Predict future (next 7 days)
last_date = df['date'].max()
future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
future_ordinals = [date.toordinal() for date in future_dates]
future_temps = model.predict(np.array(future_ordinals).reshape(-1, 1))

print("\nFuture Temperature Predictions:")
for date, temp in zip(future_dates, future_temps):
    print(f"{date.strftime('%Y-%m-%d')}: {temp:.2f}°C")

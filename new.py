import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# 1. Konfigurasi Streamlit
st.title("Real-Time Stock Prediction with LSTM")
st.write("Masukkan simbol saham dan klik 'Prediksi'.")

# Input pengguna
symbol = st.text_input("Masukkan simbol saham (contoh: AAPL, TSLA):", value="AAPL")
predict_days = st.slider("Berapa hari prediksi ke depan?", min_value=1, max_value=30, value=7)

# Tombol prediksi
if st.button("Prediksi"):
    st.write(f"Menampilkan prediksi untuk {symbol} selama {predict_days} hari ke depan.")
    
    # 2. Ambil data dari yfinance
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # Data 2 tahun terakhir
    st.write("Mengunduh data saham...")
    data = yf.download(symbol, start=start_date, end=end_date)

    if data.empty:
        st.error("Tidak ada data untuk simbol saham ini. Coba simbol lain.")
    else:
        st.write(f"Data berhasil diunduh. {len(data)} data poin ditemukan.")
        st.line_chart(data['Close'], width=700, height=300)

        # 3. Persiapan Data
        st.write("Memproses data...")
        close_prices = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        # Data training
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]

        # Membuat dataset LSTM
        def create_dataset(dataset, look_back=60):
            X, y = [], []
            for i in range(len(dataset) - look_back):
                X.append(dataset[i:i + look_back, 0])
                y.append(dataset[i + look_back, 0])
            return np.array(X), np.array(y)

        look_back = 60
        X_train, y_train = create_dataset(train_data, look_back)
        X_test, y_test = create_dataset(test_data, look_back)

        # Ubah dimensi agar sesuai dengan input LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # 4. Bangun dan Latih Model LSTM
        st.write("Melatih model LSTM...")
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=0)

        # 5. Prediksi
        st.write("Melakukan prediksi...")
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)

        # Visualisasi hasil
        st.write("Visualisasi prediksi...")
        train_data_plot = scaler.inverse_transform(train_data)
        valid_data_plot = scaler.inverse_transform(test_data[look_back:])
        pred_data_plot = predictions

        valid = data.iloc[train_size + look_back:]
        valid['Predictions'] = pred_data_plot

        st.line_chart(valid[['Close', 'Predictions']], width=700, height=300)

        # Prediksi masa depan
        st.write("Prediksi data masa depan...")
        future_data = scaled_data[-look_back:]
        future_preds = []
        for _ in range(predict_days):
            future_input = future_data[-look_back:].reshape(1, look_back, 1)
            pred = model.predict(future_input)
            future_preds.append(pred[0, 0])
            future_data = np.append(future_data, pred[0, 0]).reshape(-1, 1)

        future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
        future_dates = [end_date + timedelta(days=i) for i in range(1, predict_days + 1)]
        future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_preds.flatten()})
        st.write(future_df)
        st.line_chart(future_df.set_index('Date'))

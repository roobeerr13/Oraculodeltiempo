from __future__ import annotations
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import joblib


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(repo_root, "data")
    outputs_dir = os.path.join(repo_root, "outputs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    raw_txt = os.path.join(data_dir, "household_power_consumption.txt")
    if not os.path.exists(raw_txt):
        print(f"Raw data file not found at {raw_txt}. Please download and extract the dataset into the data/ folder.")
        sys.exit(2)

    print("Loading data (this may take a bit)...")
    df = pd.read_csv(
        raw_txt,
        sep=';',
        header=0,
        na_values='?',
        low_memory=False,
        dtype=str,
    )

    # Combine Date and Time and parse
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
    df = df.drop(columns=['Date', 'Time'])
    df = df.set_index('Datetime')

    # Convert numeric columns
    numeric_cols = []
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col].str.strip(), errors='coerce')
            numeric_cols.append(col)
        except Exception:
            # keep non-numeric as-is
            pass

    print(f"Numeric columns: {numeric_cols}")

    # Identify missing values
    missing_before = df[numeric_cols].isna().sum().sum()
    print(f"Total missing numeric entries before imputation: {missing_before}")

    # Forward fill missing values
    df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
    missing_after = df[numeric_cols].isna().sum().sum()
    print(f"Total missing numeric entries after ffill: {missing_after}")

    # Resample to daily frequency (sum)
    df_daily = df.resample('D').sum()
    daily_csv = os.path.join(outputs_dir, 'daily_consumption.csv')
    df_daily.to_csv(daily_csv)
    print(f"Saved daily aggregated CSV to: {daily_csv}")

    # Visualization: plot Global_active_power if exists, otherwise the first numeric column
    plot_col = 'Global_active_power' if 'Global_active_power' in df_daily.columns else (numeric_cols[0] if numeric_cols else None)
    if plot_col is not None:
        plt.figure(figsize=(12, 5))
        sns.lineplot(data=df_daily, x=df_daily.index, y=plot_col)
        plt.title(f"Daily {plot_col} (sum)")
        plt.xlabel('Date')
        plt.ylabel(plot_col)
        plt.tight_layout()
        plot_path = os.path.join(outputs_dir, 'daily_consumption.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved daily consumption plot to: {plot_path}")
    else:
        print("No numeric columns found to plot.")

    # Normalization: apply MinMaxScaler to numeric columns
    if numeric_cols:
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_daily_numeric = df_daily[numeric_cols].fillna(0)
        scaled = scaler.fit_transform(df_daily_numeric)
        df_scaled = pd.DataFrame(scaled, index=df_daily_numeric.index, columns=df_daily_numeric.columns)
        scaled_csv = os.path.join(outputs_dir, 'daily_consumption_scaled.csv')
        df_scaled.to_csv(scaled_csv)
        scaler_path = os.path.join(outputs_dir, 'scaler.joblib')
        joblib.dump(scaler, scaler_path)
        print(f"Saved scaled CSV to: {scaled_csv}")
        print(f"Saved fitted MinMaxScaler to: {scaler_path}")
    else:
        print("No numeric columns available for scaling.")


if __name__ == '__main__':
    main()

from automate_Nabila_Febriyanti_Valentin import preprocess_data
import pandas as pd

def main():
    X_train, X_test, y_train, y_test = preprocess_data("healthcare-dataset-stroke-data.csv")

    # Simpan ke CSV
    pd.DataFrame(X_train).to_csv('X_train.csv', index=False)
    pd.DataFrame(X_test).to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

    # Optional print untuk cek di log GitHub Actions
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train distribution:\n{y_train.value_counts()}")

if __name__ == "__main__":
    main()

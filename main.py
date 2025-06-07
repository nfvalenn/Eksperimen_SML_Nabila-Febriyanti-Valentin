from automate_Nabila_Febriyanti_Valentin import preprocess_data

X_train, X_test, y_train, y_test = preprocess_data("healthcare-dataset-stroke-data.csv")

print(f"X_train shape: {X_train.shape}")
print(f"y_train distribution:\n{y_train.value_counts()}")

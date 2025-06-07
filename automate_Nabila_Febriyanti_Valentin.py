import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(filepath):
    # Load data
    df = pd.read_csv(filepath)
    
    # Isi nilai kosong pada kolom 'bmi' dengan median
    df['bmi'].fillna(df['bmi'].median(), inplace=True)

    df.drop(columns=['id'], inplace=True)

    # Label Encoding untuk kolom kategorikal
    label_enc_cols = df.select_dtypes(include=['object'])
    le = LabelEncoder()
    for col in label_enc_cols:
        df[col] = le.fit_transform(df[col])

    # Pisahkan fitur dan target
    X = df.drop(columns=['stroke'])
    y = df['stroke']

    # Normalisasi data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test

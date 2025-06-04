import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

# fetch dataset 
chronic_kidney_disease = fetch_ucirepo(id=336) 
  
# data (as pandas dataframes) 
X = chronic_kidney_disease.data.features 
y = chronic_kidney_disease.data.targets 

# Veriyi birleştir (X ve y'yi tek DataFrame'de birleştir)
df = pd.concat([X, y], axis=1)

print("Orijinal hedef değişken değerleri:")
print("Benzersiz değerler:", df['class'].unique())
print("Değer sayıları:")
print(df['class'].value_counts())
print("\nEksik değerler var mı?")
print(df['class'].isnull().sum())

print("\n" + "="*50 + "\n")

# Eksik verileri doldur
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col].fillna(df[col].mean(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("Eksik veriler doldurulduktan sonra:")
print("Benzersiz değerler:", df['class'].unique())
print("Değer sayıları:")
print(df['class'].value_counts())

print("\n" + "="*50 + "\n")

# LabelEncoder öncesi kontrol
print("LabelEncoder uygulamadan önce:")
target_column = df['class']
print("Benzersiz değerler:", target_column.unique())
print("Değer sayıları:")
print(target_column.value_counts())

# LabelEncoder uygula
le = LabelEncoder()
encoded_target = le.fit_transform(target_column.astype(str))

print("\nLabelEncoder uygulandıktan sonra:")
print("Benzersiz değerler:", np.unique(encoded_target))
print("Sınıf etiketleri:", le.classes_)
print("Encoding mapping:")
for i, class_name in enumerate(le.classes_):
    print(f"{class_name} -> {i}") 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from ucimlrepo import fetch_ucirepo

# fetch dataset 
chronic_kidney_disease = fetch_ucirepo(id=336) 
  
# data (as pandas dataframes) 
X = chronic_kidney_disease.data.features 
y = chronic_kidney_disease.data.targets 
  
# metadata 
print("Dataset Metadata:")
print(chronic_kidney_disease.metadata) 
print("\n" + "="*50 + "\n")
  
# variable information 
print("Variable Information:")
print(chronic_kidney_disease.variables)
print("\n" + "="*50 + "\n")

# Veriyi birleştir (X ve y'yi tek DataFrame'de birleştir)
df = pd.concat([X, y], axis=1)

# Veri hakkında bilgi
print("Dataset Shape:", df.shape)
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
print("\n" + "="*50 + "\n")

# Eksik verileri kontrol et
print("Missing values:")
print(df.isnull().sum())
print("\n" + "="*50 + "\n")

# Eksik verileri doldur
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col].fillna(df[col].mean(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Kategorik verileri etiketle
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col].astype(str))

# Özellikler ve hedef değişken
X = df.drop(df.columns[-1], axis=1)  # Son sütun hedef değişken
y = df.iloc[:, -1]  # Son sütun

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("Target classes:", np.unique(y))
print("Number of unique classes:", len(np.unique(y)))

# Eğitim ve test bölmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar ağacı sınıflandırıcısı
clf = DecisionTreeClassifier(random_state=42, max_depth=10)  # Görselleştirme için derinlik sınırlandı
clf.fit(X_train, y_train)

# Karar ağacı görselleştirme - class_names'i otomatik olarak belirle
unique_classes = np.unique(y)
if len(unique_classes) == 2:
    class_names = ["Not CKD", "CKD"]
else:
    class_names = [f"Class_{i}" for i in unique_classes]

plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=class_names, filled=True, max_depth=3)
plt.title("Decision Tree (max_depth=3 for visualization)")
plt.show()

# Karar ağacı tahmini ve değerlendirmesi
y_pred = clf.predict(X_test)
print("\n" + "="*50)
print("DECISION TREE RESULTS")
print("="*50)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Diğer modellerle karşılaştırma
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel='linear')
}

print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.3f}")

# Özellik önem derecelerini göster
print("\n" + "="*50)
print("FEATURE IMPORTANCE (Random Forest)")
print("="*50)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))

# Özellik önem grafiği
plt.figure(figsize=(10, 8))
plt.barh(feature_importance.head(10)['feature'], feature_importance.head(10)['importance'])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

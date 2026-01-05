import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Veriler yüklenir
user_data = pd.read_excel("Dataset1.xlsx", decimal=",")
campaign_data = pd.read_excel("Dataset2.xlsx", decimal=",")

# Tablolar CATEGORY_ID değerine göre birleştirilir
merged_data = pd.merge(user_data, campaign_data, on='CATEGORY_ID', how='left')

# Eksik değerler kampanyadan yararlanmayan kişiler için 0 ile doldurulur
merged_data['CAMPAIGN_ID'] = merged_data['CAMPAIGN_ID'].fillna(0)
merged_data['MAX_CASHBACK'] = merged_data['MAX_CASHBACK'].fillna(0)
merged_data['MIN_AMOUNT'] = merged_data['MIN_AMOUNT'].fillna(0)
merged_data['CASHBACK_RATE'] = merged_data['CASHBACK_RATE'].fillna(0)

# Hedef sütunu target ve features olarak ayrılır
features = merged_data[['CUS_AGE', 'CATEGORY_ID', 'TUTAR', 'CASHBACK_AMT', 'MAX_CASHBACK', 'MIN_AMOUNT', 'CASHBACK_RATE']]
target = merged_data['CASHBACK_STATUS']

# Veriyi %80 eğitim ve %20 test olarak ayrılır
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Lojistik Regresyon modeli oluşturulur
model = LogisticRegression(max_iter=500)

# Veriyi ölçeklendirilir
scaler = StandardScaler()

# Eğitim verisini ölçeklendirilir
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modeli eğitim verisi ile eğitilir
model.fit(X_train_scaled, y_train)

# Test verisi üzerinde tahmin yapılır
y_pred = model.predict(X_test_scaled)

# Doğruluk skoru çıktısı alınır
print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))

# Yeni modeli test etmek için yeni kampanya bilgilerini eklenir
new_campaign_data = {
    'CATEGORY_ID': 6,
    'MIN_AMOUNT': 100,
    'MAX_CASHBACK': 100,
    'CASHBACK_RATE': 0.2
}

# Yeni kampanya bilgileri veriye eklenir
merged_data['CATEGORY_ID'] = new_campaign_data['CATEGORY_ID']
merged_data['MIN_AMOUNT'] = new_campaign_data['MIN_AMOUNT']
merged_data['MAX_CASHBACK'] = new_campaign_data['MAX_CASHBACK']
merged_data['CASHBACK_RATE'] = new_campaign_data['CASHBACK_RATE']

# Özellikler ve hedef sütunlarını yeniden ayrılır
features = merged_data[['CUS_AGE', 'CATEGORY_ID', 'AMOUNT', 'CASHBACK_AMT', 'MAX_CASHBACK', 'MIN_AMOUNT', 'CASHBACK_RATE']]
target = merged_data['CASHBACK_STATUS']

# Tahminler kullanıcı bazında gerçekleştirilir
merged_data['PREDICTION'] = model.predict(scaler.transform(features))

# Unique ID üzerinden gruplama yaparak, sadece katılacak kullanıcıları sayılır
unique_users_qualifying = merged_data.groupby('UNIQUE_ID')['PREDICTION'].max()

# Katılacak kullanıcı sayısını bulalım
users_who_qualify_count = (unique_users_qualifying == 1).sum()

# Sonucu yazdıralım
print(f"Kampanyaya katılacak kişi sayısı: {users_who_qualify_count}")

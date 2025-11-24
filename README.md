# ⭐ Weather Humidity Prediction (Nem Tahmini – Makine Öğrenmesi)

Bu proje, **1972–2025 yılları arasında Guwahati bölgesine ait günlük hava durumu verilerini** kullanarak **günlük nem oranını (humidity)** tahmin eden bir makine öğrenimi modelini içerir.

##  Kurulum (Installation)

###  1. Depoyu Klonlayın

```bash
git clone https://github.com/nurgull6872/machinelearninghumidityprediction.git
cd machinelearninghumidityprediction
```
###  2. Gerekli Kütüphaneleri Yükleyin

```bash
pip install -r requirements.txt
```

Eğer `requirements.txt` yoksa aşağıdaki paketleri yüklemek yeterlidir:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### 3. Veri Setini Kontrol Edin

CSV dosyası aşağıdaki konumda olmalıdır:

```
data/guwahati_weather_1972_2025.csv
```

### 4. Notebook veya Kod Dosyasını Çalıştırın

####  Jupyter Notebook

```bash
jupyter notebook
```
Açılan arayüzden:

```
code.ipynb
```
dosyasını çalıştırabilirsiniz.

### Not: Model Dosyası Repoda Yok

GitHub 100 MB sınırı nedeniyle:

```
humidity_model.pkl
```

dosyası depoya yüklenmemiştir.  
Modeli yeniden eğitmek için notebook içindeki eğitim hücresini çalıştırmanız yeterlidir.




*Proje amacı:*
- 50+ yıllık hava verisini işlemek  
- Nem ile ilişkili en anlamlı meteorolojik özellikleri çıkarmak  
- Farklı regresyon algoritmalarını karşılaştırmak  
- En iyi modeli seçmek → **Random Forest**  
- Pipeline şeklinde çalışan bir tahmin sistemi kurmak


##  1. Kullanılan Veri Seti

**Dosya:** `guwahati_weather_1972_2025.csv`  
**Satır:** ~19.000  
**Kolon:** 30+

| name     | datetime   | temp | tempmax | tempmin | dew  | humidity | windspeed | visibility | solarradiation |
| -------- | ---------- | ---- | ------- | ------- | ---- | -------- | --------- | ---------- | -------------- |
| guwahati | 1973-01-01 | 16.6 | 23.1    | 11.1    | 10.2 | 69.7     | 2.9       | 12.3       | NaN            |
| guwahati | 1973-01-02 | 16.2 | 22.1    | 10.1    | 12.0 | 78.7     | 3.1       | 13.5       | NaN            |

##  2. Veri Temizleme ve Dönüştürme İşlemleri

###  Tarih Formatı Dönüştürüldü (Year–Month–Day):

```python
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
```
Makine öğrenmesi tarih verisini gün, ay ve yıl olmak üzere bilgileri ayrı özellik olarak çıkarır.Tarih ile çalışmaz tarih nesnesine çevirir.
Yukarıdaki kod satırları da bunu sağlar.

###  Eksik Değer Analizi

```python
missing_table = df.isnull().sum().to_frame("Eksik Değer Sayısı")
```
Yukarıdaki kod satırı isnull() dan true dönen değerleri sum ile sütun bazlı hesaplar ve "eksik değer sayısı" isimli tabloya çevirir.

###  Hedef Değişkende Eksik Olan Satırlar Silindi

```python
df = df.dropna(subset=["humidity"]).copy()
```
Dropna kullanılarak target hedef değişkende bulunan eksik değerleri siler eğer silinmezse makine öğrenmesi yani eğitilmesi kısmında sorunlar olabilir.

## 3. Hedef Değişken ve Özellikler

### **Target:**  
`humidity`

### **Features:**

```python
features = [
    "temp","tempmax","tempmin","dew",
    "windspeed","visibility","precip",
    "solarradiation","uvindex",
    "year","month","day"
]
```

Bu değişkenler nemle anlamlı ilişki taşıdığı için seçildi.

## 4. Korelasyon Analizi

```python
sns.heatmap(df[features + ["humidity"]].corr(), cmap="coolwarm")
```
Bu kod satırı ile yalnızca features ve target değişken kullanılarak bir dataframe oluşturuldu ve korelasyon matrisi oluşturulup ısı haritası oluşturuldu.

**Çıkan Sonuçlar:**

- `dew` → nem ile en güçlü pozitif ilişki  
- `tempmax`, `tempmin` → ters korelasyon  
- `windspeed` → düşük ilişki
Modelin ısı haritası aşağıda göründüğü gibidir.
kırmızıya yakın renkler **pozitif** maviye yakın renkler **negatif** korelasyonu temsil etmektedir.

![ısıharitasi](image-2.png)

## 5. Model Seçimi: Neden Random Forest?

Nem tahmini için test edilen regresyon modelleri:

| Model                  | Durum       | Açıklama |
|-----------------------|-------------|----------|
| Linear Regression     | ❌ Zayıf     | Nem ilişkisi doğrusal değil |
| Polynomial Regression | ❌ Aşırı öğrenme | 12+ özellikte patlıyor |
| SVR                   | ❌ Çok yavaş | 19k satır için uygun değil |
| Decision Tree         | ❌ Kararsız | Tek ağaç yüksek varyanslı |
| Random Forest         | ✅ En uygun | Yüksek doğruluk + düşük hata |


Öncelikle **Linear Regression** ele alındığında, bu modelin temel varsayımı bağımsız değişkenlerle hedef değişken arasında doğrusal bir ilişki bulunmasıdır. Ancak nem, sıcaklık, çiy noktası, rüzgar hızı gibi meteorolojik değişkenler çoğunlukla karmaşık ve doğrusal olmayan ilişkiler gösterdiğinden Linear Regression bu veri üzerinde anlamlı bir performans sergileyememiştir. Model, veri setinin gerçek doğasını yakalamakta yetersiz kalmış, düşük R² skorları ve yüksek hata değerleri üretmiştir.

**Polynomial Regression**, teoride doğrusal olmayan ilişkileri yakalayabilmesi sayesinde bir alternatif olarak değerlendirilmiştir. Fakat bu yaklaşım 12’den fazla özelliğe sahip veri setlerinde hızla karmaşık hale gelir; özellikle çok boyutlu meteorolojik verilerde küçük gürültülerin bile model tarafından aşırı hassas şekilde öğrenilmesi, gerçek test performansını düşürmektedir. Bu nedenle Polynomial Regression pratik bir çözüm olmaktan uzak kalmıştır.

Bir diğer seçenek olan **SVR (Support Vector Regression)**, teorik olarak güçlü bir regresyon yöntemidir. Kernel yapısı sayesinde doğrusal olmayan ilişkileri başarıyla modelleyebilir. Ancak bu yöntem, özellikle büyük veri kümelerinde yüksek hesaplama maliyetiyle bilinir. Kullanılan veri seti yaklaşık 19.000 satır içerdiğinden, SVR’nin eğitim süresi ciddi derecede uzamakta ve modelin optimize edilmesi hem zaman hem de işlemci gücü açısından verimsiz hale gelmektedir. Bu nedenle SVR uygulamada kullanılabilir olmamıştır.

**Decision Tree Regressor**, yapısal olarak kolay anlaşılabilir ve hızlı çalışan bir algoritmadır; fakat tek bir karar ağacına dayalı olması onu oldukça kararsız kılar. Veri içinde küçük değişiklikler yapıldığında bile modelin tamamen farklı karar yapıları üretmesi mümkündür. Ayrıca tek ağaç modelleri genellikle yüksek varyansa sahiptir, veriyi aşırı derecede ezberleyebilir ve genelleme performansında büyük düşüşler görülür. Bu sebeplerle Decision Tree, büyük ve gürültülü meteorolojik veri setleri için güvenilir bir seçenek değildir.


### Random Forest neden bu projede en iyisi?

- Çoklu ağaç yapısı sayesinde **kararsızlığı azaltır**  
- **Gürültülü veriye dayanıklıdır**  
- **Doğrusal olmayan** ilişkileri çok iyi öğrenir  
- Karmaşık özellik etkileşimlerini yakalayabilir  
- Büyük veri setlerinde hızlı ve stabildir  

Bu özellikler nedeniyle Random Forest açık ara en dengeli ve başarılı model olmuştur.

##  6. Model Eğitimi (Pipeline Yapısı)

```python
model = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("rf", RandomForestRegressor(
        n_estimators=200,
        max_depth=16,
        random_state=42
    ))
])

model.fit(X_train, y_train)
```

### Neden Pipeline?

- Eksik değerler otomatik doldurulur  
- Tüm aşamalar **tek adımda** uygulanır  
- Eğitim ve tahmin sürecinde tutarlılık sağlar  
- Kod daha temiz ve profesyonel hale gelir
- 
## 7. Model Performansı

- **MAE:** ≈ 4  
- **R²:** 0.8+

### MAE  
Nem 0–100 aralığında olduğu için MAE ≈ 4 oldukça iyi bir sonuçtur.

### R²  
0.8 üzeri → model veri varyansının çoğunu açıklayabiliyor.

## 8. Gerçek vs Tahmin Görselleştirmesi

Model performansı görsel olarak:

![gercekvstahmin](image.png)


## 🌟 9. Özellik Önem Analizi

```python
importance_table = pd.DataFrame({
    "Özellik": features,
    "Önem": rf.feature_importances_
})
```

**En önemli değişkenler:**

- dew (çiy noktası)  
- tempmin / tempmax  
- solarradiation  
- visibility  
- windspeed

![enonemliozellik](image-1.png)

##  10. Örnek Tahmin

```python
sample = pd.DataFrame({
    "temp": 25,
    "tempmax": 40,
    "tempmin": 50,
    "dew": 18.5,
    "windspeed": 50,
    "visibility": 8,
    "precip": 0.1,
    "solarradiation": 150,
    "uvindex": 6,
    "year": 2024,
    "month": 7,
    "day": 12
})
```

**Tahmin edilen nem:** **73.4 %**


##  11. Modeli Kaydetme

```python
joblib.dump(model, "humidity_model.pkl")
```


## 🧾 12. Sonuç

- 50 yıllık hava durumu verisi işlendi  
- Nem tahmini için anlamlı özellikler çıkarıldı  
- Farklı modeller karşılaştırıldı  
- Random Forest en yüksek başarıyı gösterdi  
- Ortalama ±4 hata ile güçlü bir tahmin performansı elde edildi  

Bu model, gelecekte hava tahmini, bölgesel iklim araştırmaları, kuraklık analizi gibi çalışmalarda kolayca genişletilebilir bir temel sunmaktadır.

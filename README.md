⭐  Weather Humidity Prediction 

**1. Projenin Amacı**

50+ yıllık hava durumu verisini işlemek

Nem tahmini için anlamlı özellikler çıkarma

Farklı regresyon modellerini karşılaştırma

En iyi sonucu veren modeli seçme

Regression için yazılan kodun içeriğini inceleme

Tahmin yapabilen bir makine öğrenimi pipeline’ı oluşturma

**2. Kullanılan Veri Seti**

***Veri seti:***

guwahati_weather_1972_2025.csv


Veri setinde toplam ~19.000 satır / 30+ kolon bulunmaktadır.

İlk satırlar örnek olarak:

| name     | datetime   | temp | tempmax | tempmin | dew  | humidity | windspeed | visibility | solarradiation |
| -------- | ---------- | ---- | ------- | ------- | ---- | -------- | --------- | ---------- | -------------- |
| guwahati | 1973-01-01 | 16.6 | 23.1    | 11.1    | 10.2 | 69.7     | 2.9       | 12.3       | NaN            |
| guwahati | 1973-01-02 | 16.2 | 22.1    | 10.1    | 12.0 | 78.7     | 3.1       | 13.5       | NaN            |



**3. Veri Temizleme & Dönüştürme İşlemleri**

Model öncesi veri üzerinde yapılan işlemler:

* Tarih formatının dönüştürülmesi *
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day

* Eksik değerlerin tespiti *

missing_table = df.isnull().sum().to_frame("Eksik Değer Sayısı")

* Hedef değişkeni için temizleme *

Nem değeri olmayan satırlar silindi:

df = df.dropna(subset=["humidity"]).copy()

**4. Hedef Değişken (Target) ve Özellikler (Features)**

Target (Tahmin edilen):

***humidity***


Kullanılan Özellikler:

features = [
    "temp","tempmax","tempmin","dew",
    "windspeed","visibility","precip",
    "solarradiation","uvindex",
    "year","month","day"
]


Bu değişkenler nem ile anlamlı ilişkiye sahip olduğu için seçildi.

**5. Kolonlar Arasındaki İlişkiler (Korelasyon Analizi)**

Korelasyon ısı haritası:

sns.heatmap(df[features + ["humidity"]].corr(), cmap="coolwarm")


Isı haritası ile:

dew (çiy noktası) → nem ile en güçlü pozitif ilişki

tempmax & tempmin → nem ile ters korelasyon

windspeed → düşük ilişki

Gözlemlendi.

Modelin ısı haritası aşağıda göründüğü gibidir.
![ısıharitasi](image-2.png)

**6. Model Seçimi: Neden Random Forest?**



Bu proje kapsamında nem tahmini yapmak için çeşitli regresyon algoritmaları karşılaştırmalı olarak değerlendirilmiştir. Her bir algoritmanın veri setinin yapısına, değişkenlerin ilişkilerine ve veri miktarına nasıl tepki verdiği incelenmiş; hem teorik hem de pratik performans açısından artıları ve eksileri analiz edilmiştir.

Öncelikle **Linear Regression** ele alındığında, bu modelin temel varsayımı bağımsız değişkenlerle hedef değişken arasında doğrusal bir ilişki bulunmasıdır. Ancak nem, sıcaklık, çiy noktası, rüzgar hızı gibi meteorolojik değişkenler çoğunlukla karmaşık ve doğrusal olmayan ilişkiler gösterdiğinden Linear Regression bu veri üzerinde anlamlı bir performans sergileyememiştir. Model, veri setinin gerçek doğasını yakalamakta yetersiz kalmış, düşük R² skorları ve yüksek hata değerleri üretmiştir.

Polynomial Regression, teoride doğrusal olmayan ilişkileri yakalayabilmesi sayesinde bir alternatif olarak değerlendirilmiştir. Fakat bu yaklaşım 12’den fazla özelliğe sahip veri setlerinde hızla karmaşık hale gelir; modellenen polinom derecesi arttıkça model hem hesaplama açısından ağırlaşır hem de eğitim verisini ezberlemeye başlayan ağır bir overfitting eğilimi gösterir. Özellikle çok boyutlu meteorolojik verilerde küçük gürültülerin bile model tarafından aşırı hassas şekilde öğrenilmesi, gerçek test performansını düşürmektedir. Bu nedenle Polynomial Regression pratik bir çözüm olmaktan uzak kalmıştır.

Bir diğer seçenek olan SVR (Support Vector Regression), teorik olarak güçlü bir regresyon yöntemidir. Kernel yapısı sayesinde doğrusal olmayan ilişkileri başarıyla modelleyebilir. Ancak bu yöntem, özellikle büyük veri kümelerinde yüksek hesaplama maliyetiyle bilinir. Kullanılan veri seti yaklaşık 19.000 satır içerdiğinden, SVR’nin eğitim süresi ciddi derecede uzamakta ve modelin optimize edilmesi hem zaman hem de işlemci gücü açısından verimsiz hale gelmektedir. Bu nedenle SVR uygulamada kullanılabilir olmamıştır.

Decision Tree Regressor, yapısal olarak kolay anlaşılabilir ve hızlı çalışan bir algoritmadır; fakat tek bir karar ağacına dayalı olması onu oldukça kararsız kılar. Veri içinde küçük değişiklikler yapıldığında bile modelin tamamen farklı karar yapıları üretmesi mümkündür. Ayrıca tek ağaç modelleri genellikle yüksek varyansa sahiptir, veriyi aşırı derecede ezberleyebilir ve genelleme performansında büyük düşüşler görülür. Bu sebeplerle Decision Tree, büyük ve gürültülü meteorolojik veri setleri için güvenilir bir seçenek değildir.

Tüm bu değerlendirmeler sonucunda, Random Forest Regressor açık ara en başarılı ve en uygun algoritma olarak öne çıkmıştır. Random Forest, birden fazla karar ağacının birlikte çalıştığı bir topluluk (ensemble) yöntemidir. Bu yapı:

-tek bir ağacın kararsızlığını ortadan kaldırır,

-gürültüye karşı dayanıklılık sağlar,

-doğrusal olmayan ilişkileri çok iyi öğrenir,

-aşırı öğrenmeyi azaltır,

karmaşık değişken etkileşimlerini yakalayabilir.

Ayrıca model, veri setinin büyüklüğüne oldukça uygundur; paralel ağaç yapıları sayesinde hem hızlı hem de istikrarlı sonuçlar üretir. Random Forest’ın doğrudan özellik önem değerlerini sunabilmesi, modelin hangi değişkenlerden daha çok etkilendiğini anlamayı kolaylaştırmış ve yorumlanabilirlik açısından da ek bir avantaj sağlamıştır.

Sonuç olarak, yapılan tüm modelleme ve değerlendirme çalışmalarında Random Forest, hem yüksek doğruluk oranı hem de düşük hata metrikleriyle en iyi performansı gösteren yöntem olmuştur. Veri setinin yapısı, değişkenlerin ilişkileri ve problem türü göz önüne alındığında, nem tahmini için en mantıklı, en dengeli ve en güvenilir seçenek Random Forest olarak belirlenmiştir

**7. Model Eğitimi**
model = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("rf", RandomForestRegressor(
        n_estimators=200,
        max_depth=16,
        random_state=42
    ))
])

model.fit(X_train, y_train)

***Neden Pipeline?***

Eksik değerler otomatik doldurulur

Model tek adımda eğitilir

Kod daha temizdir

**8. Model Performansı**
MAE: 4.x
R²: 0.8+ 


*** MAE (Mean Absolute Error)***
Modelin ortalama hata miktarıdır.
Nem oranı 0–100 arası olduğundan MAE ≈ 4 oldukça iyi bir performanstır.

***R² Score***
Model başarısını ölçer.
0.8 üzeri değer → model veriyi iyi açıklıyor demektir.


** 8.Gerçek vs Tahmin Tablosu **

Model performansı görsel olarak:
![gercekvstahmin](image.png)


**9. Özellik Önem Analizi**

Random Forest, hangi değişkenin ne kadar etkili olduğunu gösterir.

importance_table = pd.DataFrame({
    "Özellik": features,
    "Önem": rf.feature_importances_
})


En önemli özellikler:

dew (çiy noktası) – en güçlü belirleyici

tempmin / tempmax

solarradiation

visibility

windspeed

Bu sonuçlar meteorolojik olarak da tamamen mantıklıdır.

![enonemliozellik](image-1.png)

**10. Örnek Tahmin**

Örnek bir günün verisi:

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


Tahmin:

 Tahmin edilen nem: 73.4%


**11. Model Kaydetme**
joblib.dump(model, "humidity_model.pkl")


Sonraki projelerde doğrudan kullanılabilir.


**12. Sonuç (Final Model Değerlendirmesi)**

Bu çalışmada:

50 yıllık hava durumu verisi işlendi,

Nem tahmini için anlamlı özellikler belirlendi,

Farklı regresyon yöntemleri analiz edildi,

Random Forest en iyi sonuçları verdi,

Model ortalama ±4 puan hata ile başarılı tahmin yapabiliyor.

Bu model, gelecekte hava durumu tahmini, bölgesel analizler veya iklim çalışmaları için genişletilebilir bir temel sunmaktadır.
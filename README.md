#  Weather Humidity Prediction (Nem Tahmini – Makine Öğrenmesi)

Bu proje, **1972–2025 yılları arasında Guwahati bölgesine ait günlük hava durumu verilerini** kullanarak **günlük nem oranını (humidity)** tahmin eden bir makine öğrenimi modelini içerir.


*Proje amacı:*
- 50+ yıllık hava verisini işlemek  
- Nem ile ilişkili en anlamlı meteorolojik özellikleri çıkarmak  
- Farklı regresyon algoritmalarını karşılaştırmak  
- En iyi modeli seçmek  **Random Forest**  
- Pipeline şeklinde çalışan bir tahmin sistemi kurmak


##  Kullanılan Veri Seti

**Dosya:** guwahati_weather_1972_2025.csv 
**Satır:** ~19.000  
**Kolon:** 30+

| name     | datetime   | temp | tempmax | tempmin | dew  | humidity | windspeed | visibility | solarradiation |
| -------- | ---------- | ---- | ------- | ------- | ---- | -------- | --------- | ---------- | -------------- |
| guwahati | 1973-01-01 | 16.6 | 23.1    | 11.1    | 10.2 | 69.7     | 2.9       | 12.3       | NaN            |
| guwahati | 1973-01-02 | 16.2 | 22.1    | 10.1    | 12.0 | 78.7     | 3.1       | 13.5       | NaN            |

# KOD ANALİZİ
### veri yükleme ve okuma
![veriyukleme](images/image-3.jpg)

Bu adımda önce çalıştığım klasörü os.getcwd() ile aldım, sonra os.path.join() kullanarak veri dosyasının tam yolunu birleştirdim. Böylece Python CSV dosyasının nerede olduğunu net şekilde biliyor oldu. Ardından pd.read_csv() komutuyla dosyayı içeri aktardım ve DataFrame’e dönüştürdüm. Verinin tablo şeklinde gelmesi sayesinde hem sütunlara hem de değerlere rahatça erişebildim. Son olarak da df.head(10) ile ilk 10 satıra baktım bu tamamen veri doğru yüklenmiş mi diye kontrol etmek için yaptığım bir adım oldu.

### tarih formatı dönüştürülmesi
![tarihformatıdön](images/image-5.jpg)

Verideki tarih sütunu yazı şeklindeydi yani bu da demek oluyor ki model için uygun değildi bu yüzden pd.to_datetime() ile bunu gerçek bir tarih formatına çevirdim ayrıca makine öğrenmesi tarih nesnesi ile çalışır. Sonra bu sütundan ayrı ayrı yıl, ay ve gün bilgilerini çıkartıp yeni sütunlar oluşturdum. Bu adımı modelin daha güçlü ve düzgün öğrenebilmesi için yaptım.

### eksik değer tablosu
![tarihformatıdön](images/image-6.jpg)

df.isnull().sum() komutu ile hangi sütunda kaç tane eksik değer var diye kontrol yaptım ve bu eksik değerleri toplayıp bir frame yani bu kod satırında yaptığım tabloya dönüştürdüm. Bu sayede çıktıda eksik değerleri görebildim.

### target Değişkeni ve features Belirleme
![tarihformatıdön](images/image-7.jpg)

Tahmin edeceğim değişken humidity olduğu için onu target isimli bir değişken sabite atadım ve bu sütunda eksik olan satırları dropna fonksyonundan yararlanarak sildim. Daha sonra modele girdi olarak vereceğim sütunları bir features listesi adından bir liste halinde topladım (sıcaklık, rüzgâr hızı, maksimum sıcaklık , yıl, ay, gün vb.). Bunlara özellik deniyor ve model bu bilgileri kullanarak nemi tahmin edecek ve bu özellikler değişirse her birinin değiştikleri orana uygun olarak yeni bir değer tahmin edecek bu nedenle bu kod satırını yazdım. 

### test-train ayrıştırması
![tarihformatıdön](images/image-8.jpg)

Bu kod satırında veri setini test ve train olmak üzere iki parçaya böldüm. X değişkeni modele girdi olarak vereceğim tüm bağımsız değişkenleri yani features listesinde belirttiğim değişkenleri içeriyor aynı zamanda y ise modelin tahmin etmeye çalıştığı hedef değişkendir yani humidity değeridir çoğu durumda odaklandığımız değişkendir bağımlı değişken.
Bu ayırmayı yapmamın sebebiise her makine öğrenmesi modelinin aynı mantıkla çalışıyor olmasıdır.Makine öğrenmesi modeli x içindeki değerleri kullanarak y yi tahmin eder,öğrenir.Eğer x ve y olarak ayrılmazsa model neyin girdi neyin çıktı olduğunu öğrenemez.

### korelasyon ısı haritası
![tarihformatıdön](images/image-9.jpg)


Bu adımda Seaborn kullanarak özellikler ile nem arasındaki ilişkiyi görselleştirdim bu grafik sonucunda özellikler ve target değişken arasında nasıl bir uyum veya ilişki var görselleştirebiliyoruz. Örneğin sıcaklık veya maksimum sıcaklık nemle güçlü bir ilişkiye sahipse bunu ısı haritasında kırmızıya yakın bir renk ile görebilyorum aynı şekilde de zayıfsa mavi tonlarındaki renkler ile görebiliyorum. Bu adımın var olmasının sebebi model kurmadan önce veriyi tanımak ve hangi özelliklerin gerçekten faydalı olduğunu anlamaktır sonuç çıktısında bunu hemen kolay bir şekilde analiz edebiliyorum. 

*MODELİMİN ISI HARİTASI*

![ısıharitasi](image-2.png)

### random forest modeli kurulumu
![tarihformatıdön](images/image-10.jpg)

Bu adımda bir pipeline oluşturdum e bu pipeline yapısı aynı zamanda bana farklı görevlerini yerine getirebilmemi sağladı. İlk olarak SimpleImputer ile eksik değerleri genel değerlerin ortalaması ile doldurdum çünkü modeller eksik veriyle çalışamaz çalışsa bile hatalı sonuç üretir. Ardından RandomForestRegressor modelini  pipelinea rf adıyla ekledim  yani aslında Random Forestı doğrudan pipeline zincirinin içine bağlamış oldum. Böylece model çalışırken önce eksik değerler otomatik olarak dolduruluyor, ardından çıkış olarak Random Forest modeli çalıştırılarak tahmin işlemi yapılıyor. Bu kod satırları ile pipelne yapısını kullanıp aynı anda hem bir taraftan preprocessing hem de model eklenimini yapmış oldum.

### performans ölçümü
![tarihformatıdön](images/image-11.jpg)

Bu adımda iki önemli performans belirleyici faktör kullandım. biri mae(Ortalama Mutlak Hata) diğeri ise r^2 (Başarı Skoru) mae modelin ne kadar hata yaptığını sayı olarak gösterir başarı skoru ise 0-1 arasında bir skor üreterek modelin veriyi ne kadar iyi açıklayıp açıklamadığını gösterir. Denediğim diğer regresyon modelleri ile kıyaslamamda da bu performans ölçütlerini kullandım ve çıktılara göre random forest modeline karar verdim.

### 
![tarihformatıdön](images/image-12.jpg)

Bu bölümde ilk olarak modelimin ne kadar doğru tahmin yaptığını görsel olarak incelemek için bir scatter grafiği çizdim. Bu grafikte test setindeki gerçek nem değerlerini yatay eksene, modelin tahmin ettiği nem değerlerini ise dikey eksene yerleştirdim. Böylece her bir nokta aslında “model bu değeri böyle tahmin etmiş” anlamına geliyor. Grafiğe ayrıca kırmızı bir çizgi ekledim; bu çizgi modelin birebir doğru tahmin yaptığı ideal durumu temsil ediyor. Noktaların bu çizgiye yakın olması modelimin gerçeğe ne kadar yaklaştığını hızlı bir şekilde görmemi sağlıyor. Sonrasında modelin hangi özelliklere daha çok önem verdiğini öğrenmek için Random Forest’ın feature_importances_ değerlerini kullandım. Pipeline içindeki Random Forest modelini çıkartıp tüm özelliklerin önem skorlarını bir tabloya dönüştürdüm ve en önemliden en aza doğru sıraladım. Bu tabloyu oluşturmaktaki amacım, modelin nemi tahmin ederken hangi değişkenleri daha etkili bulduğunu anlamaktı. Örneğin model “dew” veya “tempmin” özelliklerine daha fazla önem veriyorsa bu, nem ile bu değişkenler arasında güçlü bir ilişki olduğu anlamına geliyor. Son olarak bu tabloyu bar grafik olarak çizdim çünkü görsel bir grafik sayesinde hangi özelliğin daha önemli olduğunu çok daha net ve anlaşılır bir şekilde görebiliyorum. Bu adımların hepsi benim için önemli çünkü hem modelin ne kadar iyi tahmin yaptığını görüyorum hem de modelin nasıl karar verdiğini, yani “iç mantığını” daha iyi çözmüş oluyorum. Bu sayede hem performansı hem de modelin çalışma şeklini çok daha rahat değerlendirebiliyorum.

*GERÇEK VS TAHMİN*

![gercekvstahmin](image.png)


*ÖZELLİK ÖNEM TABLOSU*
**En önemli değişkenler:**

- dew 
- tempmin / tempmax  
- solarradiation  
- visibility  
- windspeed

![enonemliozellik](image-1.png)






## MODEL SEÇİMİ

Kullandığım veri setimde regresyon modeli seçmek için farklı modeller denedim ve bu modelelr arasından en iyi seçebilmek için de testler uyguladım bu testler modelin veri ile ne kadar uyuştuğunu ne kadar iyi tahminler yaptığını iyi tahmin olup olmadığını da grafikler çizerek ölçmeye çalıştım kullandığım testyöntemleri ilk olarak mae ve r^2 dir bu ölçümler bir modelin ne kadar hata yapıtğını ve veri ile uyumluluğunu ölçen genelgeçer kuramlardır ve devamında da her bir regresyn modeli için ayrı ayrı tahmin vs gerçek grafikleri çizdirdim bu grafikte olması gereken değerlerve modelin tahmin ettiği değerlerin birbirleri ile uyuşup uyumadığını ölçtüm.

**Linear Regression** 
Bu model bilindiği üzere genellikle doğrusal ilişkileri ele alan bir modeldir yukarıda bahsettiğim test içeriklerini uyguladığımda ise aşağıdaki sonuçları aldım 
![gercekvstahmin](regressiontestimages/image-15.png)
![gercekvstahmin](regressiontestimages/image-14.png)

*MAE: 0.94* bu sonuç bu modelin gerçek değerden ortalama 0.94 kadar saptığını gösteriyor kullandığım veri setindekş metorolojik gibi karışık verilerde ise bu gibi hatalar oldukça küçük kabıl ediliyor yani bu değe bize modelin gerçek değerlere yakın sonuçlar veridğini gösteriyor
*R²: 0.976* bu değer ise varyans değeri olarak da biliniyor ve veri değişiminin açklanabilirliğini gösteriyor bu sonuca bakıldığında anlıyoruz ki bu veri modeli verinin yaklaşık %97 sini açıklayabiliyor yani gerçek değerlerle tahminlerin uyuşmasını bir grafik üzerinde gösteriyor ortadaki kırmızı çizgi net tahmin yani doğru değerleri gösteriyor bu kırmızı doğrusal çizgiye uygunluk, yakınlık ne kadar çoksa veri tahmininin o kadar doğru olacağını biliyoruz bu grafikte de linear regresyon modelinin doğruluğuna bir kez daha emin oldum çoğunlukla kırmızı çizgi etrafında birleşim ile ayrıca aynı eksran görüntülerinde grafik de çizdirdim ve bu grafik gerçek vs tahmin ve bu da doğrusal veriler için özellikle kullanışlı olan bu model için benim açımdan beklenmedik sonuç oldu.


Bu değer, modelin verideki değişimin yaklaşık %97’sini açıkladığını gösteriyor. Başka bir deyişle, nemin nasıl arttığını ve azaldığını model neredeyse tamamen doğru şekilde öğrenmiş.

Grafik sonuçları da bunu destekliyor. Çizdiğim Gerçek – Tahmin grafiğinde, mavi noktaların çoğu kırmızı “mükemmel tahmin” çizgisinin hemen yakınında duruyor. Bu da modelin veriyi çok iyi yakaladığını ve büyük sapmalar olmadığını gösteriyor.

Genel olarak:
Linear Regression bu veri setinde oldukça başarılı sonuç verdi. Hem hata değerleri çok düşük hem de tahminler grafik üzerinde neredeyse gerçek çizgiyle çakışıyor. Yine de bu model sadece doğrusal ilişkileri öğrenebildiği için, veri daha karmaşık davranışlar içeriyorsa başka regresyon modellerini de denemek gerekiyor.


Öncelikle ele alındığında, bu modelin temel varsayımı bağımsız değişkenlerle hedef değişken arasında doğrusal bir ilişki bulunmasıdır. Ancak nem, sıcaklık, çiy noktası, rüzgar hızı gibi meteorolojik değişkenler çoğunlukla karmaşık ve doğrusal olmayan ilişkiler gösterdiğinden Linear Regression bu veri üzerinde anlamlı bir performans sergileyememiştir. Model, veri setinin gerçek doğasını yakalamakta yetersiz kalmış, düşük R² skorları ve yüksek hata değerleri üretmiştir.

**Polynomial Regression**, teoride doğrusal olmayan ilişkileri yakalayabilmesi sayesinde bir alternatif olarak değerlendirilmiştir. Fakat bu yaklaşım 12’den fazla özelliğe sahip veri setlerinde hızla karmaşık hale gelir ve özellikle çok boyutlu meteorolojik verilerde küçük gürültülerin bile model tarafından aşırı hassas şekilde öğrenilmesi, gerçek test performansını düşürmüştür. Bu nedenle Polynomial Regression pratik bir çözüm olmaktan uzak kalmıştır.

Bir diğer seçenek olan **SVR (Support Vector Regression)**, teorik olarak güçlü bir regresyon yöntemidir. Kernel yapısı sayesinde doğrusal olmayan ilişkileri başarıyla modelleyebilir. Ancak bu yöntem, özellikle büyük veri kümelerinde yüksek hesaplama maliyetiyle bilinir. Kullandığım veri setim yaklaşık 19.000 satır içerdiğinden, SVR’nin eğitim süresi ciddi derecede uzamakta ve modelin optimize edilmesi hem zaman hem de işlemci gücü açısından verimsiz hale gelmiştir. Bu yöntem pratk olarak hiç uygun olamamıştır.Bu nedenle SVR uygulamada kullanılabilir olmamıştır.

**Decision Tree Regressor**, yapısal olarak kolay anlaşılabilir ve hızlı çalışan bir algoritmadır fakat tek bir karar ağacına dayalı olması onu oldukça kararsız kılar. Veri içinde küçük değişiklikler yapıldığında bile modelin tamamen farklı karar yapıları üretmesi mümkündür. Ayrıca tek ağaç modelleri genellikle yüksek varyansa sahiptir bu da demek oluyor ki veriyi aşırı derecede ezberleyebilir ve genelleme performansında büyük düşüşler görülür. Bu sebeplerle Decision Tree, büyük ve gürültülü meteorolojik veri setleri için tıpkı benim kullandığım veri seti gibi veri setleri için güvenilir bir seçenek değildir.


### Random Forest neden bu projede en iyisi?

- Çoklu ağaç yapısı sayesinde **kararsızlığı azaltır**  
- **Gürültülü veriye yani eksik ve düzgün olmayan verilere dayanıklıdır**  
- **Doğrusal olmayan** ilişkileri çok iyi öğrenir  
- Karmaşık özellik etkileşimlerini yakalayabilir  
- Büyük veri setlerinde hızlı ve stabildir  

Bu özellikler nedeniyle üstte kullanılan ve denenen diğer algoritmalardan farklı olarak Random Forest açık ara en dengeli ve başarılı model olmuştur.



##  10. Örnek Tahmini 
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
Modelin bir deger tahmin edebilmesi için default olarak bazı değerler verildi ve işlem yaptırıldı. Bu değerler değişkenlik gösterdiğinde modelin tahmin değeri değişecektir.
**Tahmin edilen nem:** **73.4 %**


##  11. Modeli Kaydetme


joblib.dump(model, "humidity_model.pkl")

Modeli kaydetmemiz durumunda sonraki kullanımlarda kolaylık sağlanacaktır. Ama bu kaydedilmiş model github 100 mb sınırından dolayı yüklenemedi. Proje çalıştırıldığında yüklenecektir.

## 12. Sonuç

- 50 yıllık hava durumu verisi işlendi  
- Nem tahmini için anlamlı özellikler çıkarıldı  
- Farklı modeller karşılaştırıldı  
- Random Forest en yüksek başarıyı gösterdi  
- Ortalama +4 -4 hata ile güçlü bir tahmin performansı elde edildi  


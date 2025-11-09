import os   # dosya ve klasörleri okuyabilmesi için import ettim 
import cv2  # video okuma, yüz kırpma, renk dönüştürme(BGR den RGB) için
import numpy as np   # real = 0, fake = 1 etiketlemesini dizi olarak tutabilmesi için numpy kullandım
import pandas as pd  # daha önce excell dosyaları  ile çalışmıştım dataframe yapısını bildiğim için pandas kullandım
from mtcnn import MTCNN   
from sklearn.model_selection import train_test_split   
from sklearn.preprocessing import StandardScaler       # verileri ölçeklendirmek için
from imblearn.over_sampling import SMOTE               # dengesiz veri setlerini dengelemek için
from keras.applications.resnet50 import ResNet50, preprocess_input   # resnet50 modeli hazır eğitilmiş bir model . Bunu görüntü işlemek içiin hazır olarak kullandım
from keras.preprocessing.image import img_to_array                   # görsel vektörleri numpy dizisine çevirmek için import ettim 
from keras.models import Sequential, Model                           # video gerçek mi sahte mi sınıflandırması için modeli tanımlarken kullandım bunu hazır kullanmıyorum 
#                                                                     #kendim eğitiyorum
from keras.layers import Dense                              # tam bağlı katmanlar(dense) 
from keras.optimizers import Adam                                    # modelin öğrenmesini sağlayan optimizasyon algoritması

# resnet50 modelinin tanımlama
base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")  

# resnet50 modelini ImageNet ağırlıklarıyla yüklüyor 
# bunu kullanma amacım: sistemim çok iyi değil sıfırdan eğitsem yüz, insan,köşe,kenar tespiti çok uzun sürecekti ve muhtemelen bilgisayarım kaldırmayacaktı
# bu model 1.2 milyon görsel ile eğitilmiş o yüzden elimdeki az veri ile yüksek doğruluk oranı yakalamak için kullandım (yani yüz  tespiti için transfer learning yaptım)
model_cnn = Model(inputs=base_model.input, outputs=base_model.output)  

detector = MTCNN()  
# MTCNN yüz tespiti nesnesi oluşturuyorum


# Videoyu embeddinge yani videoyu özellik vektörüne çevirmek için kullandım
def video_to_embedding(video_path, max_frames=15):   # Bir videodan maksimum 15 kare alarak özellik vektörü çıkarıyor 
    #                                                 #bunu 5 ile denedim hızlı sonuç veriyor ama doğruluk oranı çok düşükte kalıyor 
    #                                                  25 ise çok yavaş ve yine doğruluk düşüyor o yüzden 15 yaptım
    cap = cv2.VideoCapture(video_path)   # Videoyu kare kare okumak için VideoCapture nesnesini tanımladım
    frame_count = 0   # İşlenen kare sayısını tutar
    embeddings = []   # Her kareden elde edilen yüz embeddingleri bu listeye eklenecek

    while True:   # Kareleri tek tek okuma döngüsü
        ret, frame = cap.read()   # 1 kare okur
        if not ret or frame_count >= max_frames:   # Video bittiğinde veya 15 kareye ulaşıldığında döngü durur
            break
        frame_count += 1   # Okunan kare sayısını bir artırır 15 e kadar tekrarlar

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # OpenCV kareleri BGR formatında okur, MTCNN RGB ister
        faces = detector.detect_faces(rgb)             # Karedeki yüzleri tespit eder

        if faces:   # Eğer karede yüz bulunduysa
            x, y, w, h = faces[0]['box']   # İlk bulunan yüzün koordinatları alınır
            face = rgb[y:y+h, x:x+w] # Bu koordinatlara göre yüz bölgesi kesilir

            try:
                face = cv2.resize(face, (224,224))          # ResNet50 giriş boyutuna uygun şekilde yeniden boyutlandır bu 224x224 olacak
                face_array = img_to_array(face)             # Görüntüyü numpy dizisine dönüştür
                face_array = np.expand_dims(face_array, axis=0)   
                face_array = preprocess_input(face_array)         

                embedding = model_cnn.predict(face_array, verbose=0)  # resnet50den embeddingi çıkar
                embeddings.append(embedding[0])  
            except:
                continue   # Herhangi bir hata olursa hatalı kareyi atla örnek veriyorum yüz kadrajın dışına çıktı o kareyi atlıyor

    cap.release()   # videoyu atlıyor

    if embeddings:   # eğer en az bir karede yüz bulunduysa
        return np.mean(embeddings, axis=0)   # tüm karelerin ortalama embeddingini döndür
    else:
        return None   # hiç yüz bulunmazsa none döndür


# Dataset oluşturma

video_folder = "videos/"  # ana klasörün altında "real" ve "fake" klasörleri var 400er adet video koydum
data = []    # Her video için embeddingleri tutar
labels = []  # Her video için sınıf etiketlerini tutar (real=0, fake=1)

for label in ["real", "fake"]:   # Gerçek ve sahte videoları sırayla işliyor
    folder = os.path.join(video_folder, label)   
    for file in os.listdir(folder):              # klasördeki her dosyayı dolaş
        if file.endswith(".mp4"):                # sadece .mp4 dosyalarını al
            path = os.path.join(folder, file)    
            emb = video_to_embedding(path)       # videodan embedding çıkar
            if emb is not None:                  # eğer yüz bulunduysa
                data.append(emb)                 # embeddingi veri listesine ekle
                labels.append(1 if label=="fake" else 0)  # fake=1, real=0 etiketi ekle

# DataFrame oluşturma
X = np.array(data)    # embeddingler
y = np.array(labels)  # etiketler
df = pd.DataFrame(X)  # embeddingleri tablo haline getir
df['video'] = y       #etiketleri tabloya ekle


# Train-Test Split + SMOTE + Scaling  TensorFlow un yapay sinir ağı modeli için veriyi hazırlama ksmı

X = df.drop('video', axis=1)   # Etiket sütununu Xten çıkar
y = df['video']                # Etiketleri y ye ata

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Veriyi %70 eğitim, %30 test olarak böl 
# (random_state =42 için yazılımcı şakası diyorlar genelde böyle alınır diyorlar ben 20 ve 60 denedim 42 de daha fazla doğruluk yakaladığım için öyle bıraktım)

smote = SMOTE(random_state=1)   # iki veriyi de artırırken kalsörde eşit sayıda video tuttum ama yine de tedbir amaçlı SMOTE kullandım
#SMOTEu kaldırmayı düşünüyorum çünkü yavaşlatıyor  
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
# eğitim verisinde azınlık sınıfını artırarak sınıf dengesini sağlıyor ben ikisinide hep eşit sayıda tuttuğum için bu kısmı kaldırıcam muhtemelen

scaler = StandardScaler()   
X_train_scaled = scaler.fit_transform(X_train_smote)   # eğitim verisini fit+transform ile ölçeklendir
X_test_scaled = scaler.transform(X_test)               # test verisini sadece transform ile dönüştür


# Yapay Sinir Ağı (MLP)

print('Model: Yapay Sinir Ağı')
model = Sequential()   # Sıralı model tanımlanır
model.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))  # giriş katmanı
model.add(Dense(64, activation='relu'))  
model.add(Dense(32, activation='relu'))  
model.add(Dense(1, activation='sigmoid'))  # Çıkış katmanı (0,1 şeklinde sınıflandırma için sigmoid)

optimizer = Adam(learning_rate=0.001)  
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# Modeli eğitme
model.fit(X_train_scaled, y_train_smote, epochs=70, batch_size=16, verbose=10, validation_data=(X_test_scaled, y_test))
# 70 epoch boyunca modeli eğitir, her epochta doğrulama verisiyle kontrol eder terminale yazdırır


#modelin performansını değerlendiriyor
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {accuracy:.4f}")  # doğruluk oranını yazdır


#Projenin durumu hakkında kısa bir özet
#Kaggle da deepfake detection challenge yarışmasında kullanılan bir dataseti indirdim (https://www.kaggle.com/c/deepfake-detection-challenge/data)
#içerisinde gerçek ve sahte videolar var 400 er adet video kullandım
#Videolardan MTCNN ile yüz tespiti yaptım, tespit edilen yüzleri ResNet50 modelinden geçirdim ve embeddinglerini aldım
# ResNet50 modelini kullanma sebebim elimdeki az veri ve kısıtlı sistem ile doğru bilgi elde edebilmekti

#Elde ettiğim embeddingleri kullanarak yapay sinir ağı modeli ile gerçek mi sahte mi sınıflandırması yaptım
#TensorFlow Keras kütüphanesini kullanarak modeli tanımladım ve eğittim
#Tek tek deneme yanılma yolu ile parametreler ile oynayıp doğruluk oranlarını test ettim bazı denemelerim şu şekilde:
# 27 video ,25 epoch 8 batchsize,3 katman 64 relu,32 relu,1 output,max frame = 5 ile yuzde 56.25 doğruluk oranı
# 140 video ,25 epoch 8 batchsize,3 katman 64 relu,32 relu,1 output,max frame = 5 ile yuzde 60.49 doğruluk oranı
# 140 video ,25 epoch 16 batchsize,3 katman 64 relu,32 relu,1 output,max frame = 15 ile yuzde 67.86 doğruluk oranı
# 140 video ,25 epoch 8 batchsize,3 katman 64 relu,32 relu,1 output,max frame = 25 ile yuzde 64.29 doğruluk oranı
# 252 video ,25 epoch 8 batchsize,3 katman 64 relu,32 relu,1 output,max frame = 15 ile yuzde 70 doğruluk oranı
# 252 video ,40 epoch 16 batchsize,4 katman 128,relu ,64 relu,32 relu,1 output,max frame = 15 ile yuzde 76.51 doğruluk oranı
# 400 video ,40 epoch 16 batchsize,4 katman 128,relu ,64 relu,32 relu,1 output,max frame = 15 ile yuzde 70.42 doğruluk oranı
# en son ChatGPT ye doğruluk oranımı nasıl artırabileceğimi sordum Dropout fonksiyonu kullanmamı söyledi
# Teorik bilgide eksik olduğum için henüz dropout eklemedim 
# SMOTE u kaldırmayı düşünüyorum çünkü zaten eşit sayıda fake ve real video kullanıyorum 
#bu projeyi yazarken yararlandığım kurslar şunlar :
# https://www.udemy.com/course/python-ve-yapay-zekaya-giris-101/?kw=yapay+zekaya+giri%C5%9F&src=sac
# https://www.udemy.com/course/yapay-zeka/

 
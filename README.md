# Görüntü Sınıflandırma Projesi

Merhaba! Bu proje, Keras kütüphanesi kullanılarak görüntü sınıflandırma modeli oluşturmayı ve eğitmeyi amaçlamaktadır. Farklı önceden eğitilmiş model mimarilerini kullanarak çeşitli deneyler yapabilir ve görüntü sınıflandırma problemini çözmek için en uygun modeli bulabilirsiniz.

## Nasıl Başlayabilirim?

1. İlk adım olarak projeyi bilgisayarınıza klonlayın:

    ```bash
    git clone https://github.com/kutaybaskurt/image-classification-project.git
    ```

2. Ardından projenin dizinine gidin:

    ```bash
    cd image-classification-project
    ```

3. Gereksinimleri yüklemek için aşağıdaki komutu çalıştırın:

    ```bash
    pip install -r requirements.txt
    ```

4. Model oluşturma ve eğitme işlemi için `main.py` dosyasını kullanabilirsiniz.

5. Oluşturulan modelin performansını değerlendirmek için aşağıdaki komutu çalıştırın:

    ```bash
    python evaluate_model.py
    ```

## Veri Yapısı

Proje, görüntü verisini `/Dataset` dizinindeki klasörler içinde organize edilmiş olarak kullanmaktadır. Her sınıf için ayrı bir klasör bulunmalıdır.

## Desteklenen Önceden Eğitilmiş Modeller

- ResNet50
- InceptionV3
- InceptionResNetV2
- Xception
- EfficientNetB7
- EfficientNetV2L

Proje, bu modelleri kullanarak görüntü sınıflandırma modeli oluşturmayı destekler.

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Daha fazla bilgi için [LICENSE](LICENSE) dosyasına bakabilirsiniz.

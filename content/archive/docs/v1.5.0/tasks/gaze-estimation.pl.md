---
title: Estymacja spojrzenia
seo_title: Estymacja spojrzenia w LibreYOLO
description: >-
  Estymuj kąty pitch i yaw spojrzenia dla każdej twarzy w LibreYOLO. Przewiduj z
  Pythona lub CLI, odczytuj kąty w radianach i eksportuj głowicę spojrzenia do
  ONNX.
lead: >-
  Estymacja spojrzenia zwraca kierunek patrzenia dla każdej twarzy na obrazie.
  LibreYOLO modeluje ją jako zadanie dwuetapowe: najpierw działa detektor
  twarzy, a głowica spojrzenia odczytuje pitch i yaw z każdego zwróconego
  wycinka twarzy.
keywords:
  - estymacja spojrzenia python
  - eye tracking
  - pitch yaw spojrzenia
  - L2CS-Net
  - kierunek patrzenia
  - pozycja głowy
  - libreyolo gaze
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Jeśli nie podano face_detector, predykcja używa dołączonego detektora

        # OpenCV, więc poza checkpointem niczego nie trzeba pobierać.

        model = LibreYOLO("LibreL2CSr50.pt")

        result = model(SAMPLE_IMAGE)


        gaze = result.gaze

        print(gaze.pitch, gaze.yaw)              # radiany, jeden wiersz na
        twarz

        print(gaze.pitch_deg, gaze.yaw_deg)      # te same kąty w stopniach

        print(gaze.direction_3d)                 # (N, 3) wektory jednostkowe
    - label: CLI
      language: bash
      code: >
        # W przeciwieństwie do ścieżki Pythona CLI nie ma automatycznego
        rozwiązania

        # zapasowego. Modele spojrzenia wymagają jawnego detektora twarzy,
        którym musi

        # być detektor LibreYOLO zwracający ramki twarzy.

        libreyolo predict model=LibreL2CSr50.pt source=photo.jpg
        face_detector=face-detector.pt save=True
    - label: Wybór źródła twarzy
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreL2CSr50.pt")

        # Przekaż głowicy spojrzenia ramki z wcześniej uruchomionego detektora.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])

        # Albo podaj nazwę jednego z dołączonych detektorów.
        result = model(SAMPLE_IMAGE, face_detector="yunet")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreL2CSr50.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreL2CSr50.pt format=onnx
source_hash: 22aa3c3d87b0c730
---

## Definicja

Estymacja spojrzenia zwraca dwa kąty dla każdej twarzy. `result.gaze` jest
strukturą `Gaze` o kształcie `(N, 2)`, w której kolumna 0 oznacza pitch, a
kolumna 1 yaw, w radianach. Wiersze są wyrównane z `result.boxes`, czyli ramkami
wykrytych twarzy. Konwencja pochodzi z L2CS-Net. Dodatni yaw obraca kierunek
spojrzenia w lewą stronę osoby, a dodatni pitch kieruje go w dół.

Ta sama struktura udostępnia `pitch_deg` i `yaw_deg` dla stopni oraz
`direction_3d`, czyli wektor jednostkowy `(N, 3)` w układzie kamery z kolumnami
`(x, y, z)`.

Ponieważ zadanie składa się z dwóch etapów, predykcja zależy od dwóch modeli.
Twarze pominięte przez detektor nie mają wiersza spojrzenia, a źle umieszczone
ramki dają kąty z nieprawidłowo wyciętej twarzy. Kanoniczny klucz zadania to
`gaze`, a `gaze-estimation` jest do niego normalizowane.

## Modele

[L2CS-Net](/docs/models/l2cs) jest jedyną rodziną obsługującą to zadanie. Łączy
trzon ResNet z dwiema równoległymi głowicami klasyfikującymi przedziały kątów,
jedną dla pitch i jedną dla yaw, na wycinkach twarzy 448x448. Architektura
obsługuje pięć głębokości backbone, a jedna z nich, ResNet-50, ma opublikowany
checkpoint.

Wagi podlegają ograniczeniu licencyjnemu. Wytrenowano je na Gaze360, którego
licencja zezwala wyłącznie na użycie badawcze i niekomercyjne oraz zabrania
redystrybucji. Dlatego LibreYOLO nie udostępnia kopii żadnego pliku tej rodziny.
Jedyny checkpoint, który biblioteka może pobrać automatycznie, pochodzi
bezpośrednio z dystrybucji autorów w Google Drive, przez `gdown`, po wyświetleniu
warunków licencji. Przed wdrożeniem należy przeczytać stronę
[L2CS-Net](/docs/models/l2cs).

Ta ścieżka pobierania wymaga zestawu zależności `gaze`:

```bash
pip install "libreyolo[gaze]"
```

Bez niego biblioteka wyświetla instrukcje ręcznego pobierania zamiast próbować
transferu. Predykcja i eksport posiadanego checkpointu nie wymagają żadnych
dodatkowych zależności.

## Predykcja

<code-tabs name="predict" />

Źródło twarzy wybiera się na jeden z trzech sposobów. `face_boxes` przekazuje
wcześniej obliczone ramki i pomija detekcję. `face_detector` przyjmuje `"auto"`,
`"haar"`, `"yunet"`, model detekcji LibreYOLO lub zwykły obiekt wywoływalny i
może zostać ustawiony w konstruktorze albo dla pojedynczego wywołania. Jeśli w
Pythonie pozostanie nieustawiony, predykcja korzysta z dołączonego detektora
OpenCV, dzięki czemu proste wywołanie działa bez dodatkowej konfiguracji. W
OpenCV 4 jest to kaskada Haar dołączona do pakietu wheel i niewymagająca żadnego
pobierania. W OpenCV 5, z którego usunięto API Haar, używany jest YuNet,
pobierający jednorazowo mały plik modelu z katalogu OpenCV.

CLI nie korzysta z tego rozwiązania zapasowego. `libreyolo predict` odrzuca
model spojrzenia bez `face_detector=`, a wartością musi być nazwa detektora
LibreYOLO lub ścieżka checkpointu. Informacje o źródłach, streamingu i obsłudze
wyników zawiera strona [predykcji](/docs/predict).

## Trenowanie

Żadna rodzina w tym zadaniu nie jest trenowana w LibreYOLO. Funkcja
`LibreL2CS.train()` zgłasza wyjątek. Trenowanie należy przeprowadzić w projekcie
źródłowym L2CS-Net, a wynikowy słownik stanu wczytać tutaj.

## Walidacja

Walidacja względem zbiorów danych z referencyjnym spojrzeniem znajduje się poza
zakresem, a `val()` zgłasza wyjątek zamiast zwracać nieobliczone metryki. Dla
tego zadania nie ma słownika `metrics/`. Ocenę należy przeprowadzić w projekcie
źródłowym na zbiorze danych, dla którego wytrenowano checkpoint.

## Eksport

<code-tabs name="export" />

Kontrakt eksportu spojrzenia obejmuje ONNX, TorchScript, ExecuTorch, TensorRT i
OpenVINO. Bibliotekę opuszczają tylko trzon ResNet i dwie głowice przedziałów
kątowych. Graf przyjmuje wstępnie przetworzony wycinek twarzy 448x448 i zwraca
surowe logity yaw oraz pitch. Detekcja twarzy, wycinanie, softmax, wartość
oczekiwana przedziału i konwersja na kąty pozostają w Pythonie, w
`libreyolo.models.l2cs.utils`. Formaty i ich argumenty opisuje strona
[eksportu](/docs/export).

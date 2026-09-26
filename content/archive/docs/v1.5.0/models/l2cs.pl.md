---
title: L2CS-Net
families:
  - l2cs
seo_title: 'L2CS-Net: estymacja kierunku spojrzenia w LibreYOLO'
description: >-
  Używaj L2CS-Net w LibreYOLO do dwustopniowej estymacji kąta nachylenia i
  obrotu spojrzenia. Instaluj, przewiduj i eksportuj. Checkpoint Gaze360 służy
  wyłącznie do badań.
lead: >-
  L2CS-Net to dwustopniowy estymator kierunku spojrzenia: detektor twarzy
  lokalizuje twarze, a główna część ResNet z dwiema głowicami klasyfikacji
  przedziałów kątowych przewiduje nachylenie i obrót dla każdej twarzy.
  LibreYOLO udostępnia go wyłącznie do inferencji.
keywords:
  - L2CS-Net
  - estymacja kierunku spojrzenia
  - śledzenie wzroku
  - pitch yaw
  - Gaze360
  - detekcja twarzy
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Gdy nie podano face_detector, używany jest dołączony detektor twarzy
        OpenCV

        # (Haar w OpenCV 4, YuNet w OpenCV 5), dlatego przykład działa bez

        # pobierania czegokolwiek poza samym checkpointem L2CS.

        model = LibreYOLO("LibreL2CSr50.pt")

        result = model(SAMPLE_IMAGE)


        print(result.gaze.pitch, result.gaze.yaw)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreL2CSr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Źródło twarzy
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreL2CSr50.pt")

        # Przekazanie do L2CS ramek z wcześniej uruchomionego detektora.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])

        # Albo wskazanie konkretnego dołączonego detektora twarzy.
        result = model(SAMPLE_IMAGE, face_detector="yunet")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreL2CSr50.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreL2CSr50.pt format=onnx
    - label: Użycie wyeksportowanego pliku
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # Wyeksportowany graf zawiera tylko główną część ResNet i dwie głowice
        przedziałów

        # kątowych. Przyjmuje przetworzone wstępnie przycięcie twarzy 448x448 i
        zwraca surowe

        # (yaw_logits, pitch_logits), a nie zdekodowane kąty. Softmax,

        # wartość oczekiwana przedziału i konwersja na stopnie pozostają w
        Pythonie. Zobacz

        # libreyolo.models.l2cs.utils.bin_logits_to_angles.

        session = ort.InferenceSession("LibreL2CSr50.onnx")

        name = session.get_inputs()[0].name

        yaw_logits, pitch_logits = session.run(
            None, {name: np.zeros((1, 3, 448, 448), dtype=np.float32)}
        )
source_hash: 4ec43f4673b4be3e
---

## Instalacja

L2CS-Net nie wymaga dodatku do utworzenia modelu, uruchamiania predykcji ani
eksportu, jeśli checkpoint jest już dostępny.

```bash
pip install libreyolo
```

Jedyny checkpoint, który LibreYOLO może pobrać automatycznie, czyli ResNet-50
wytrenowany na Gaze360, jest pobierany przez `gdown`, a nie zwykłą kopię
lustrzaną HTTP, ponieważ znajduje się na Google Drive autora zamiast w
organizacji LibreYOLO. Ta ścieżka wymaga dodatku `gaze`:

```bash
pip install "libreyolo[gaze]"
```

Bez niego LibreYOLO wypisuje instrukcje ręcznego pobierania zamiast po cichu
kończyć działanie błędem.

## Predykcja

<code-tabs name="predict" />

L2CS-Net jest estymatorem dwustopniowym. Najpierw działa detektor twarzy, a
głowica spojrzenia odczytuje nachylenie i obrót z każdego zwróconego przycięcia
twarzy. Bez dodatkowej konfiguracji predykcja korzysta z dołączonego detektora
OpenCV, dlatego zwykłe wywołanie działa bez dodatkowego pobierania, gdy
checkpoint L2CS jest już dostępny. `face_boxes` przyjmuje ramki z wcześniej
uruchomionego detektora. `face_detector` przyjmuje `"auto"`, `"haar"`,
`"yunet"`, model detekcji LibreYOLO albo zwykły obiekt wywoływalny.
`result.gaze` zawiera nachylenie i obrót w radianach, wyrównane wiersz po wierszu
z `result.boxes`, czyli wykrytymi ramkami twarzy. Więcej informacji o źródłach,
streamingu i obsłudze wyników zawiera strona [predykcji](/docs/predict).

## Warianty

Pięć głębokości backbone współdzieli jedną rozdzielczość wejściową i przyjmuje
te same argumenty. Gaze360, zbiór danych stojący za jedynym opublikowanym
checkpointem, posłużył do wytrenowania ResNet-50. Pozostałe cztery głębokości
są obsługiwane architektonicznie, ale nie mają opublikowanych wag do wczytania.

## Eksport

<export-matrix />

<code-tabs name="export" />

## Licencja

<provenance-box>

LibreYOLO nie hostuje ani nie tworzy kopii lustrzanej żadnego checkpointu L2CS.
W przeciwieństwie do większości pozostałych rodzin na tej stronie w organizacji
LibreYOLO na Hugging Face nie ma żadnych plików tej rodziny. Jedyny checkpoint,
który biblioteka może pobrać automatycznie, pochodzi bezpośrednio z własnej
dystrybucji autora na Google Drive, jest kontrolowany przez informację o
licencji Gaze360 wyświetlaną przed rozpoczęciem transferu i nie jest kopią
„republished at huggingface.co/LibreYOLO”, którą sugeruje powyższe podsumowanie.

</provenance-box>

---
title: LibreFaceRec
families:
  - facerec
seo_title: 'LibreFaceRec: rozpoznawanie i weryfikacja twarzy'
description: >-
  Używaj LibreFaceRec w LibreYOLO do detekcji twarzy, tworzenia embeddingów i
  weryfikacji. Instaluj i przewiduj. Wagi embeddingów podlegają licencji
  Apache-2.0.
lead: >-
  LibreFaceRec to zadanie embeddingów twarzy w LibreYOLO: detektor twarzy
  lokalizuje i wyrównuje twarze, a głowica rozpoznawania tworzy znormalizowany
  metodą L2 embedding tożsamości do weryfikacji lub wyszukiwania.
keywords:
  - LibreFaceRec
  - rozpoznawanie twarzy
  - embedding twarzy
  - weryfikacja twarzy
  - ArcFace
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Nazwy librefacerec-* kierują do tej rodziny niezależnie od
        rozszerzenia

        # pliku i przy pierwszym użyciu pobierają pliki z organizacji LibreYOLO

        # na Hugging Face, razem z domyślnym detektorem twarzy.

        model = LibreYOLO("librefacerec-l.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.embeddings.data.shape)   # (N, D), normalizacja L2
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=face.jpg
    - label: Weryfikacja
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("librefacerec-l.onnx")


        # Porównuje najbardziej widoczną twarz na każdym obrazie za pomocą
        podobieństwa

        # cosinusowego ich embeddingów znormalizowanych metodą L2.

        result = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)

        print(result["similarity"], result["same_person"])
    - label: Wyszukiwanie w galerii
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        query = model("query.jpg").embeddings          # twarze na tym obrazie
        gallery = model.embed(["a.jpg", "b.jpg", "c.jpg"])   # (N_total, D)

        # Podobieństwa cosinusowe (query_faces, N_total).
        scores = query.similarity(gallery)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")
        model.export(format="onnx")
source_hash: f1a345bb96e32f12
---

## Instalacja

Głowica rozpoznawania LibreFaceRec działa przez `onnxruntime`, które nie jest
częścią instalacji podstawowej.

```bash
pip install "libreyolo[onnx]"
```

## Predykcja

<code-tabs name="predict" />

Za jednym wywołaniem kryją się dwa osobne grafy ONNX do detekcji i rozpoznawania.
Detektor twarzy lokalizuje i wyrównuje każdą twarz do kanonicznego przycięcia,
a głowica rozpoznawania zwraca dla każdej twarzy embedding znormalizowany metodą
L2. Bez dodatkowej konfiguracji `predict()` automatycznie pobiera i łączy
dołączony domyślny detektor. `face_detector` przyjmuje wywoływalny obiekt, model
detekcji LibreYOLO albo instancję `FaceDetector`. `face_boxes` całkowicie pomija
detekcję, korzystając z już dostępnych ramek. `result.embeddings` zawiera jeden
wiersz na wykrytą twarz, wyrównany z `result.boxes`. Jego metoda `.similarity()`
oblicza w jednym wywołaniu podobieństwo cosinusowe względem innego embeddingu lub
całej galerii. Aby bezpośrednio porównać dwa obrazy zamiast dwóch wcześniej
obliczonych embeddingów, `model.verify(image_a, image_b)` uruchamia detekcję i
tworzenie embeddingów dla obu obrazów, a następnie porównuje ich twarze o
najwyższej pewności. Można zastąpić głowicę dowolnym innym modelem rozpoznawania
ONNX zgodnym z konwencją ArcFace, który przyjmuje wyrównane przycięcie i zwraca
embeddingi `(N, D)`. Wystarczy przekazać ścieżkę jego pliku zamiast nazwy
`librefacerec-*`. Więcej informacji o źródłach, streamingu i obsłudze wyników
zawiera strona [predykcji](/docs/predict).

## Eksport

<export-matrix />

LibreFaceRec już opakowuje wcześniej wyeksportowany graf ONNX. Ponowny eksport
do innego formatu nie jest zaimplementowany.

## Licencja

<provenance-box>

Dołączony domyślny detektor twarzy jest drugim artefaktem podlegającym osobnej
licencji: YuNet z OpenCV Zoo, licencja MIT, prawa autorskie Shiqi Yu. Kod
architektury nie jest przenoszony z żadnego projektu. Oba grafy są używane jako
nieprzezroczyste artefakty przez `onnxruntime`, dlatego własny wrapper LibreYOLO
nie zawiera kodu zewnętrznego i w całości podlega licencji MIT.

</provenance-box>

## Cytowanie

<citation-block />

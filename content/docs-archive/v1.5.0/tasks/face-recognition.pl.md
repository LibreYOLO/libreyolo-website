---
title: Rozpoznawanie twarzy
seo_title: Rozpoznawanie twarzy w LibreYOLO
description: >-
  Wykrywaj twarze, twórz embeddingi i identyfikuj osoby w LibreYOLO. Rejestruj
  galerię, porównuj dwa obrazy i dopasowuj podobieństwem cosinusowym z Pythona
  lub CLI.
lead: >-
  Rozpoznawanie twarzy jest zadaniem embed zastosowanym do twarzy. Detektor
  lokalizuje i wyrównuje każdą twarz, głowica rozpoznawania zwraca
  znormalizowany metodą L2 wektor dla każdej twarzy, a o tożsamości decyduje
  podobieństwo cosinusowe względem zarejestrowanych referencji zamiast stałej
  listy klas.
keywords:
  - rozpoznawanie twarzy python
  - embedding twarzy
  - weryfikacja twarzy
  - galeria twarzy
  - arcface onnx
  - libreyolo embed
  - podobieństwo cosinusowe twarzy
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Nazwy librefacerec-* prowadzą do rodziny embeddingów twarzy
        niezależnie

        # od sufiksu pliku, a przy pierwszym użyciu pobierają pliki z
        organizacji

        # LibreYOLO na Hugging Face wraz z domyślnym detektorem twarzy.

        model = LibreYOLO("librefacerec-l.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)             # (N, 4) ramki twarzy

        print(result.embeddings.data.shape)  # (N, D), jeden wiersz na twarz

        print(result.embeddings.dim)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=photo.jpg
    - label: Porównanie dwóch obrazów
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("librefacerec-l.onnx")


        # Uruchamia detekcję i tworzenie embeddingu na obu obrazach, a następnie

        # porównuje twarz o największej pewności. Podobieństwo cosinusowe mieści
        się w [-1, 1].

        outcome = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)

        print(outcome["similarity"], outcome["same_person"])
    - label: Rejestracja galerii i identyfikacja
      language: python
      code: |
        from libreyolo import Gallery, LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        gallery = Gallery(model)
        gallery.enroll("ada", ["people/ada/1.jpg", "people/ada/2.jpg"])
        gallery.enroll("grace", "people/grace/1.jpg")
        gallery.save("faces.npz")

        result = model("group_photo.jpg", gallery=gallery, threshold=0.4)
        for name, score in result.identities.data:
            print(name, score)   # poniżej progu name ma wartość None
    - label: Rejestracja i identyfikacja z CLI
      language: bash
      code: >
        libreyolo enroll model=librefacerec-l.onnx source=people/
        gallery=faces.npz

        libreyolo predict model=librefacerec-l.onnx source=group_photo.jpg
        gallery=faces.npz
    - label: Własne ramki twarzy
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("librefacerec-l.onnx")

        # face_boxes całkowicie pomija detekcję; face_detector przyjmuje obiekt
        # wywoływalny, model detekcji LibreYOLO lub instancję FaceDetector.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])
        print(result.embeddings.data.shape)
source_hash: d7dfcb6f812ebb2d
---

## Definicja

Rozpoznawanie twarzy zwraca wektor dla każdej twarzy, a nie etykietę. Predykcja
przebiega w dwóch etapach. Detektor twarzy lokalizuje każdą twarz i jej pięć
punktów charakterystycznych, wycinek jest przekształcany do kanonicznego
wyrównania 112x112, a głowica rozpoznawania emituje embedding znormalizowany
metodą L2.

`result.embeddings` jest strukturą `Embeddings` o kształcie `(N, D)`, z wierszami
wyrównanymi względem `result.boxes`, dlatego wiersz `i` opisuje twarz w ramce
`i`. Ponieważ wiersze są wektorami jednostkowymi, podobieństwo cosinusowe jest
iloczynem skalarnym, a `embeddings.similarity()` oblicza je jednym wywołaniem
względem innego obiektu `Embeddings` lub całej macierzy.

Przypisanie twarzy nazwy jest osobnym etapem. `Gallery` przechowuje nazwane
wektory referencyjne. Przekazanie `gallery=` do `predict()` dołącza
`result.identities`, którego wiersze są wyrównane z embeddingami i zawierają
nazwę oraz najlepszy wskaźnik cosinusowy dla każdej twarzy. Twarz poniżej progu
dopasowania zachowuje nazwę `None`, a najbliższa nazwa poniżej progu nigdy nie
jest podstawiana.

Kanoniczny klucz zadania w bibliotece to `embed`. Wartości `face-recognition`,
`facial-recognition`, `reid` i `face` są do niego normalizowane, więc
`task="face-recognition"` i `task="embed"` wybierają to samo zadanie. Twarze są
regionalnym kształtem szerszego zadania. Strona [embeddingów](/docs/tasks/embeddings)
opisuje kształty całego obrazu i tekstu, wspólne API `Embeddings`, `Identities`
i `Gallery` oraz modele tworzące wektory bez lokalizowania czegokolwiek.

## Modele

[LibreFaceRec](/docs/models/librefacerec) jest rodziną przeznaczoną do tego
zadania. Za jednym wywołaniem stoją dwa artefakty ONNX: `librefacerec-l.onnx`,
czyli głowica rozpoznawania iResNet100 tworząca embeddingi 512-wymiarowe, oraz
`librefacerec-det.onnx`, domyślny detektor twarzy z pięcioma punktami
charakterystycznymi pochodzący z katalogu modeli OpenCV. Przy pierwszym użyciu
oba są pobierane z organizacji LibreYOLO na Hugging Face. Głowicę rozpoznawania
może zastąpić dowolny inny plik ONNX zgodny z konwencją ArcFace (wyrównane dane
wejściowe 112x112 i wyjście `(N, D)`). W tym celu należy przekazać jego ścieżkę
zamiast nazwy `librefacerec-*`.

Klucz zadania `embed` ma szerszy zakres niż twarze. [CLIP](/docs/models/clip),
[SigLIP2](/docs/models/siglip2) i [DINOv2](/docs/models/dinov2) również obsługują
`task="embed"` i zwracają jeden wektor całego obrazu. Służy to do wyszukiwania
obrazów, a nie ustalania tożsamości twarzy. Korzystają ze wspólnego API
`Gallery` i `Embeddings`, więc poniższy przepływ rejestracji i dopasowania ma do
nich zastosowanie, ale nie wykrywają ani nie wyrównują twarzy.

Głowica rozpoznawania działa przez `onnxruntime`, którego instalacja bazowa nie
zawiera:

```bash
pip install "libreyolo[onnx]"
```

## Predykcja

<code-tabs name="predict" />

Bez dodatkowych ustawień funkcja `predict()` pobiera i dołącza domyślny
detektor. `face_detector` zastępuje go obiektem wywoływalnym, modelem detekcji
LibreYOLO lub instancją `FaceDetector` i może zostać ustawiony w konstruktorze
albo dla pojedynczego wywołania. `face_boxes` pomija detekcję i używa
posiadanych ramek. W CLI argument `face_detector=` przyjmuje ścieżkę do
detektora twarzy `.onnx` lub nazwę detektora LibreYOLO.

`model.verify(image_a, image_b)` jest skrótem dla dwóch obrazów. Tworzy embedding
twarzy o największej pewności na każdym obrazie i zwraca
`{"similarity", "same_person", "threshold"}`. `model.embed(sources)` zwraca
wszystkie wiersze twarzy z jednego lub kilku obrazów ułożone w pojedynczy tensor
`(N_total, D)`. Informacje o źródłach, streamingu i obsłudze wyników zawiera
strona [predykcji](/docs/predict).

## Format zbioru danych

Rejestracja odczytuje po jednym folderze na tożsamość. Nazwa folderu staje się
tożsamością, a każdy znajdujący się w nim obraz wnosi referencje dla tej nazwy:

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

Polecenie `libreyolo enroll` przechodzi przez to drzewo i zapisuje galerię
`.npz`. Istniejący plik galerii jest rozszerzany w miejscu zamiast zastępowany,
dzięki czemu tożsamości można dodawać z czasem. Galerie są wiązane z tworzącymi
je wagami za pomocą wymiaru embeddingu i odcisku pliku. Dopasowanie przy użyciu
innego modelu zgłasza wyjątek zamiast porównywać niezgodne przestrzenie wektorów.

Domyślnie każdy obraz źródłowy wnosi jeden wiersz referencyjny, pochodzący z
twarzy o największej pewności. Dzięki temu portret zawierający osoby postronne
rejestruje wyłącznie główny obiekt. Aby zapisać każdy zwrócony wiersz, przekaż
`select="all"` do `Gallery.enroll`.

## Trenowanie

Żadna rodzina w tym zadaniu nie jest trenowana w LibreYOLO.
`LibreFaceEmbedder.train()` zgłasza wyjątek. Głowicę rozpoznawania należy
wytrenować w projekcie źródłowym, wyeksportować do ONNX zgodnie z konwencją
ArcFace i wczytać plik ze ścieżki.

## Walidacja

Dla tego zadania nie ma walidatora zbioru danych, a `val()` zgłasza wyjątek
zamiast pozorować działanie. Dokładność weryfikacji mierzy się na oznaczonych
parach obrazów za pomocą `model.verify()`, przeszukując wartości `threshold`,
aby wybrać odpowiedni punkt pracy. Dokładność identyfikacji mierzy się przez
rejestrację galerii i odczytywanie `result.identities.name` oraz
`result.identities.score` na odłożonych obrazach. Nazwę `None` należy liczyć
jako odrzucenie.

## Eksport

Głowica rozpoznawania jest już grafem ONNX, więc nie ma czego konwertować.
`LibreFaceEmbedder.export()` zgłasza wyjątek. Plik `.onnx` należy wdrożyć
bezpośrednio albo wskazać go LibreYOLO, aby rodzina obsłużyła detekcję,
wyrównanie i normalizację.

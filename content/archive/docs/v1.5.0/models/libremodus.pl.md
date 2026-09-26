---
title: LibreMODUS
families:
  - libremodus
seo_title: 'LibreMODUS w LibreYOLO: analiza obrazu any-to-any'
description: >-
  Używaj LibreMODUS w LibreYOLO do estymacji głębi, normalnych, krawędzi i
  detekcji oraz łącz je za pomocą any2any(). Wyłącznie do inferencji. Wagi są
  wczytywane z EPFL-VILAB.
lead: >-
  LibreMODUS to integracja przeznaczonego wyłącznie do inferencji checkpointu
  MODUS 14B-A7B, modelu any-to-any przekształcającego jedno wejście pochodzące z
  obrazu w inne: RGB na głębię, głębię na normalne, a dowolne z nich wraz z
  frazą na ramki. LibreYOLO obsługuje cztery zadania przez standardowe API
  predykcji i szerszy zestaw przez any2any().
keywords:
  - LibreMODUS
  - MODUS
  - any-to-any
  - estymacja głębi
  - normalne powierzchni
  - detekcja krawędzi
  - detekcja referencyjna
  - EPFL VILAB
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS(size="14b-a7b", task="normal")
        result = model.predict("room.jpg")
        normals = result.normal_map.data

        model.set_task("edge")
        result = model.predict("room.jpg")
        edges = result.edges.data

        # Bez niestandardowego słownika detect dekoduje tokeny etykiet COCO
        # checkpointu do ciągłych identyfikatorów klas COCO-80.
        model.set_task("detect")
        result = model.predict("street.jpg")
        print(result.boxes.xyxy)
    - label: Grounding fraz
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS(task="detect")
        # set_classes() przełącza detekcję na grounding fraz: każda fraza
        # jest uruchamiana niezależnie i zwracana przez ten sam kontrakt Boxes.
        model.set_classes(["red bus", "cyclist"])
        result = model.predict("street.jpg", conf=0.2)
        print(result.boxes.xyxy, result.boxes.cls)
    - label: any2any()
      language: python
      code: >
        from libreyolo import LibreMODUS


        model = LibreMODUS()


        # Od jednego do trzech wejść pochodzących z obrazu (rgb, depth, normal,
        canny/edge)

        # oraz opcjonalny tekst pomocniczy, złożone w kierunku jednego celu.

        result = model.any2any(
            inputs={"rgb": "room.jpg"},
            target="normal",
            steps=10,
            cfg=2.0,
            seed=0,
        )

        normals = result.normal_map.data


        # Grounding przez any2any() wymaga wejścia tekstowego z nazwą frazy.

        result = model.any2any(
            {"rgb": "street.jpg", "text": "red bus"},
            target="grounding",
        )

        print(result.boxes.xyxy)
source_hash: 7386886d4c36ea9a
---

## Instalacja

LibreMODUS wymaga własnego dodatku, który instaluje `accelerate` do obsługi
dużego modelu wymaganej przez ten checkpoint.

```bash
pip install "libreyolo[modus]"
```

LibreYOLO nie redystrybuuje ani nie tworzy kopii lustrzanej wag MODUS. Domyślnie
wczytanie modelu `LibreMODUS` pobiera wymagane pliki bezpośrednio z
`EPFL-VILAB/MODUS` przy przypiętej rewizji Hugging Face. Nowe pobranie zawsze
wymaga własnego uwierzytelnionego konta Hugging Face użytkownika, nawet jeśli
bramka hostingu źródłowego jest tymczasowo otwarta. Należy przeczytać i
zaakceptować warunki źródłowe, a następnie się uwierzytelnić:

```bash
hf auth login
```

```python
from libreyolo import LibreMODUS

model = LibreMODUS(token="hf_...")
```

Aby uniknąć żądania sieciowego, można wskazać istniejący snapshot:

```python
model = LibreMODUS(checkpoint_path="/models/MODUS")
```

Ten katalog musi zawierać `model.safetensors`, `ae.safetensors`,
`llm_config.json`, `vit_config.json`, `tokenizer_config.json`, `vocab.json` i
`merges.txt`. Poniższa sekcja Licencja opisuje dozwolone zastosowania
checkpointu.

## Predykcja

<code-tabs name="predict" />

Standardowe API zadania obejmuje cztery zadania, z których każde jest mapowane
na jeden cel MODUS: `depth` na głębię względną (`result.depth_map`), `normal` na
normalne powierzchni (`result.normal_map`), `edge` na krawędzie w stylu Canny
(`result.edges`) oraz `detect` na ramki COCO-80 (`result.boxes`), chyba że
`set_classes()` przełączy je na grounding fraz. `set_task()` przełącza zadania
w tym samym wczytanym modelu. Opublikowana procedura używa dziesięciu kroków
próbkowania przepływu z prowadzeniem tekstowym 4.0 i prowadzeniem obrazowym 2.0.
Można je zastąpić argumentami `inference_steps=`, `inference_cfg=` i
`inference_image_cfg=` podczas tworzenia modelu.

`any2any()` udostępnia szerszą publiczną powierzchnię analizy: od jednego do
trzech wejść pochodzących z obrazu (`rgb`, `depth`, `normal`, `canny`/`edge`)
oraz opcjonalny tekst pomocniczy, złożone w kierunku dowolnego celu spośród
głębi, normalnych, krawędzi, krawędzi pochodzących z SAM, detekcji COCO lub
groundingu fraz. Wszystkie wejścia pochodzące z obrazu muszą opisywać ten sam
wyrównany obszar. LibreMODUS odrzuca niedopasowane szerokości i wysokości
zamiast niezależnie zmieniać ich rozmiar. `chain=(...)` generuje cele pośrednie
i przekazuje je z powrotem do tego samego kontekstu w ramach budżetu trenowania
checkpointu wynoszącego trzy warunki. `verify=N` (`N >= 2`) generuje N
kandydatów i zachowuje kandydata z najwyższym wynikiem ograniczonej kontroli
spójności własnej, udostępnionym jako `result.verification_score`.

`dtype="bf16"` (wartość domyślna) odpowiada precyzji opublikowanego checkpointu.
`dtype="fp8"` zapisuje kwalifikujące się wagi liniowe głównej części dekodera w
formacie E4M3 ze skalą dla każdego kanału wyjściowego, jednorazowo konwertuje je
do lokalnej pamięci podręcznej w `~/.cache/libreyolo/modus/fp8` i dekwantyzuje
do typu danych wejściowych podczas każdego mnożenia macierzy. Jest to kompromis
dotyczący pamięci, a nie dokładności na poziomie aktywacji.

Wywołania `train()`, `val()` i `export()` zgłaszają błąd. LibreMODUS służy
wyłącznie do inferencji, walidacja zbioru danych nie jest oferowana i nie ma
ścieżki eksportu ONNX, TensorRT ani TFLite. Predykcja w batchach i augmentacja
podczas testowania także nie są obsługiwane. Każde wywołanie przetwarza jeden
obraz.

## Licencja

<provenance-box>

LibreYOLO nigdzie nie hostuje ani nie tworzy kopii lustrzanej checkpointu MODUS,
w tym we własnej organizacji Hugging Face. Wczytanie zawsze pobiera przypiętą
rewizję bezpośrednio z EPFL-VILAB/MODUS albo odczytuje istniejący snapshot z
dysku wskazany przez `checkpoint_path`.

</provenance-box>

## Cytowanie

<citation-block />

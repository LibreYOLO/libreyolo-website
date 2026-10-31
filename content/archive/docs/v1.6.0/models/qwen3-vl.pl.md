---
title: Qwen3-VL
families:
  - qwen3vl
seo_title: 'Qwen3-VL w LibreYOLO: detekcja z otwartym słownikiem'
description: >-
  Qwen3-VL w LibreYOLO: zainstaluj model wizyjno-językowy Alibaba na licencji
  Apache-2.0, ustaw otwarty słownik, uruchamiaj predykcję lub czat.
lead: >-
  Qwen3-VL to model wizyjno-językowy Alibaba z natywnym groundingiem 2D.
  LibreYOLO udostępnia go jako detektor obiektów z otwartym słownikiem oraz
  zapewnia bezpośredni dostęp do swobodnego czatu: podaj listę klas do detekcji
  albo zadaj pytanie.
keywords:
  - Qwen3-VL
  - model wizyjno-językowy
  - detekcja z otwartym słownikiem
  - grounding obrazu
  - Alibaba
  - VLM
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("qwen3-vl-4b")
        model.set_classes(["forklift", "pallet", "safety vest"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Czat
      language: python
      code: >
        from libreyolo import LibreVLM, SAMPLE_IMAGE


        model = LibreVLM("qwen3-vl-4b")


        # Bezpośredni dostęp pod wygodną warstwą detekcji: dowolne pytanie,

        # nie tylko zapytanie o ramki ograniczające.

        answer = model.chat(SAMPLE_IMAGE, "How many people are wearing a safety
        vest?")

        print(answer)
source_hash: 801d97d089f1f957
---

## Instalacja

Qwen3-VL należy do poziomu VLM jako detektor w LibreYOLO, osobnej powierzchni
produktu od rodzin opartych na checkpointach, z własną fabryką. Wymaga dodatku
`vlm`.

```bash
pip install "libreyolo[vlm]"
```

## Predykcja

Przy pierwszym użyciu wagi są pobierane z Hugging Face i zapisywane w lokalnej
pamięci podręcznej. Wywołanie `LibreVLM()` bez argumentu domyślnie wybiera
Qwen3-VL-4B.

<code-tabs name="predict" />

Ta rodzina jest wczytywana przez fabrykę `LibreVLM()`, a nie `LibreYOLO()`.
Rodziny VLM nie deklarują loadera checkpointów, dlatego opisany na innych
stronach modeli wybór ścieżki na podstawie rozszerzenia pliku nie ma tu
zastosowania. `set_classes()` ustawia słownik, o którego znalezienie proszony
jest Qwen3-VL. Ustawienie jest trwałe i obowiązuje we wszystkich późniejszych
wywołaniach `predict()`/`track()`, dopóki nie zostanie ponownie zmienione. Każda
detekcja ma ten sam zastępczy wskaźnik pewności, więc filtrowanie przez `conf`
działa na zasadzie wszystko albo nic. `iou` ma wpływ na wynik dla tej rodziny:
późniejsza ramka tej samej klasy jest odrzucana, gdy nakłada się na zachowaną
ramkę powyżej progu, ponieważ generator powtarzający może zwrócić prawie
identyczne ramki jednego obiektu. W przeciwieństwie do Florence-2 i Kosmos-2,
Qwen3-VL odpowiada także na swobodne pytania przez `chat()`, ten sam bezpośredni
dostęp opisany dla fabryki `LibreVLM`. CLI LibreYOLO nie obsługuje tego poziomu.
Nie istnieje dla niego forma `libreyolo predict model=...`. Więcej informacji o
źródłach, streamingu i obsłudze wyników zawiera strona
[predykcji](/docs/predict).

## Warianty

Dostępne są trzy rozmiary: Qwen3-VL-2B-Instruct, Qwen3-VL-4B-Instruct i
Qwen3-VL-8B-Instruct, wczytywane jako `LibreVLM("qwen3-vl-2b")`,
`LibreVLM("qwen3-vl-4b")` i `LibreVLM("qwen3-vl-8b")`. Wszystkie trzy deklarują
nominalne wejście 1024 px, ale własny mechanizm smart-resize procesora Qwen
ustala rzeczywisty obszar przekazywany do sieci. Ta wartość nie jest więc stałą
rozdzielczością działania, tak jak w innych rodzinach na tej stronie. LibreYOLO
nie opublikowało benchmarku porównującego dokładność tych trzech rozmiarów.

## Trenowanie

Po zainstalowaniu `libreyolo[vlm-train]` można trenować adaptery LoRA do detekcji przez `LibreVLM("qwen3-vl-2b").train(data=...)`. Enkoder wizyjny pozostaje zamrożony; funkcja straty na walidacji wybiera najlepszy katalog checkpointu. Wznawianie stanu optymalizatora i walidacja mAP detekcji nie są obsługiwane. Wartości domyślne i instrukcje ponownego wczytywania opisano w sekcji [dostrajania VLM](/docs/train/vlm-fine-tuning).

## Licencja

<provenance-box></provenance-box>

## Cytowanie

<citation-block />

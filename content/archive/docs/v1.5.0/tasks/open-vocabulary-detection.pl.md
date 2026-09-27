---
title: Detekcja z otwartym słownikiem
seo_title: Detekcja z otwartym słownikiem w LibreYOLO
description: >-
  Wykrywaj w LibreYOLO obiekty ze słownika tekstowego. Wczytaj Grounding DINO,
  OWLv2, OMDet-Turbo lub OV-DEIM przez LibreOpenVocab i ustaw klasy podczas
  działania programu.
lead: >-
  Detekcja z otwartym słownikiem zastępuje stałą listę klas checkpointu słowami
  wybieranymi podczas wywołania. W LibreYOLO nie jest to osobne zadanie: jest to
  zadanie detect obsługiwane przez oddzielną warstwę modeli, wczytywaną przez
  fabrykę LibreOpenVocab zamiast LibreYOLO.
keywords:
  - detekcja z otwartym słownikiem
  - zero shot object detection
  - open set detection
  - grounding dino python
  - owlv2
  - omdet turbo
  - detekcja z promptem tekstowym
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-t")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        print(result.names)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Zmiana słownika
      language: python
      code: >
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE


        model = LibreOpenVocab("owlv2-b16")


        # set_classes zachowuje stan aż do kolejnego wywołania tej metody.

        # Etykiety po zamianie na małe litery i usunięciu przedimków muszą być
        unikatowe.

        model.set_classes(["a red backpack", "traffic cone"])

        result = model.predict(SAMPLE_IMAGE)


        model.set_classes(["bicycle wheel"])

        result = model.predict(SAMPLE_IMAGE)
    - label: Próg tekstowy Grounding DINO
      language: python
      code: >
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE


        model = LibreOpenVocab("grounding-dino-b")

        model.set_classes(["remote control", "school bus"])


        # conf filtruje według wyniku ramki, a text_threshold według wyniku
        tokenu

        # zdekodowanej frazy. Bez ustawienia oba domyślnie wynoszą 0.25. Tylko

        # DINO akceptuje text_threshold; pozostałe modele zgłaszają błąd.

        result = model.predict(SAMPLE_IMAGE, conf=0.25, text_threshold=0.3)
source_hash: 17197cf4d80f3d6f
---

## Definicja

Detekcja z otwartym słownikiem zwraca zwykłe obiekty `Results` detekcji: ramki,
wartości pewności i indeksy klas, przy czym `result.names` mapuje te indeksy z
powrotem na podane ciągi znaków. Zmienia się źródło listy klas. Konwencjonalny
detektor jest trenowany na stałym zestawie kategorii i nie może zwrócić
kategorii spoza niego. Te modele przyjmują słownik jako tekst podczas
inferencji, więc `set_classes(["forklift", "safety cone"])` wystarcza, aby te
pozycje stały się klasami.

LibreYOLO nie ma klucza zadania `open-vocabulary`. Modele te deklarują
`SUPPORTED_TASKS = ("detect",)` jak każdy inny detektor. Odróżnia je ścieżka
wczytywania. Są migawkami Hugging Face, a nie checkpointami state-dict
LibreYOLO, dlatego nie trafiają do fabryki `LibreYOLO()` i tworzy się je przez
`LibreOpenVocab()`. Ta fabryka jest równorzędna z `LibreSAM()` i `LibreVLM()`, a
nie zastępuje `LibreYOLO()`.

Wyniki są rzeczywistymi wynikami detekcji, a nie wygenerowanym podpisem
analizowanym po fakcie. Każda rodzina ocenia obszary obrazu względem embeddingu
tekstowego każdego promptu.

## Modele

Warstwę tworzą cztery rodziny, wszystkie wyłącznie do predykcji. Każdą z nich
można wczytać przez alias za pomocą `LibreOpenVocab`.

[Grounding DINO](/docs/models/grounding-dino) od IDEA Research jest dostępny w
rozmiarach `t` i `b`. To domyślna rodzina tej warstwy i jedyna, która akceptuje
`text_threshold`, czyli drugi próg dla wyniku tokenu zdekodowanej frazy.

[OWLv2](/docs/models/owlv2) od Google Research jest dostępny w rozmiarach `b16`
i `l14`. Ocenia obszary obrazu względem embeddingów tekstowych z enkodera w
stylu CLIP.

[OMDet-Turbo](/docs/models/omdet-turbo) od Om AI Lab jest dostępny w jednym
rozmiarze `t`. Oddziela embeddingi klas od promptu zadania językowego. Jest to
jedyna rodzina na tej stronie, która tłumi nakładające się ramki we własnym
przetwarzaniu końcowym, dlatego respektuje `iou=`.

[OV-DEIM](/docs/models/ov-deim), dostępny w rozmiarach `s`, `m` i `l`, jest
detektorem w stylu DETR, który dopasowuje zapytania dekodera do embeddingów
tekstowych z dołączonego modułu tekstowego MobileCLIP. Stosuje dopasowanie jeden
do jednego z wyborem top-K, dlatego NMS nie jest nigdzie uruchamiane.

Wagi OV-DEIM są przypadkiem z ograniczeniami w tej warstwie. Wagi detektora są
udostępniane na licencji CC BY-NC 4.0, wyłącznie do użytku niekomercyjnego.
Dołączony moduł tekstowy podlega licencji Apple Machine Learning Research Model,
wyłącznie do zastosowań badawczych. Checkpoint `l` dodaje dostrajanie backbone
DINOv3-S na licencji Meta DINOv3 License. Teksty wszystkich trzech licencji
znajdują się w repozytorium wag, a biblioteka rejestruje to samo podsumowanie
podczas rozwiązywania wag, przed zbudowaniem modelu. Przed wdrożeniem należy
przeczytać stronę [OV-DEIM](/docs/models/ov-deim).

Warstwa wymaga jednego dodatku:

```bash
pip install "libreyolo[openvocab]"
```

Obejmuje on pakiety `transformers` i `timm` dla trzech opakowanych rodzin oraz
`huggingface_hub`, `safetensors`, `regex` i `ftfy`, których OV-DEIM potrzebuje
jako natywny port.

Druga warstwa również przyjmuje słownik tekstowy. `LibreVLM()` wczytuje
generatywne modele wizyjno-językowe, takie jak
[Qwen3-VL](/docs/models/qwen3-vl) i [Florence-2](/docs/models/florence-2), oraz
przekształca ich dane wyjściowe w takie same obiekty `Results`. Udostępnia ten
sam interfejs `set_classes()`. Różnica polega na sposobie tworzenia ramek:
rodziny na tej stronie są detektorami dyskryminacyjnymi, które bezpośrednio
zwracają wyniki, natomiast warstwa VLM je generuje.

## Predykcja

<code-tabs name="predict" />

`set_classes()` przyjmuje niepustą listę ciągów znaków z etykietami i zachowuje
ją do następnego wywołania. Etykiety muszą być unikatowe po zamianie na małe
litery i usunięciu początkowych przedimków, więc `"a bus"` i `"bus"` nie mogą
współistnieć w jednym słowniku. Frazy wielowyrazowe są etykietami jak każde
inne. Każda rodzina przekształca listę we własne wejście tekstowe przed
tokenizacją, dlatego `"traffic cone"` jest innym zapytaniem niż `"cone"`.

Trzy argumenty predykcji zachowują się tutaj inaczej niż w natywnym detektorze.
`imgsz=` jest odrzucane, ponieważ zmianą rozmiaru w tych rodzinach zarządza
procesor. `augment=True` jest odrzucane, gdyż augmentacja w czasie testu nie
wchodzi w zakres tej warstwy. `iou=` ma zastosowanie tylko do rodziny, której
procesor wykonuje własne tłumienie. Tam, gdzie nic nie jest tłumione, przekazanie
tego argumentu powoduje ostrzeżenie, a wartość jest ignorowana.

Jeśli `conf` nie zostanie ustawione, przyjmuje własną wartość domyślną wczytanej
rodziny zamiast standardowej wartości 0.25 z `predict()`. Wartość domyślna nie
jest jednakowa w całej warstwie. Przy porównywaniu dwóch rodzin na tym samym
obrazie należy ustawić ją jawnie.

`track()` zgłasza błąd w całej warstwie. Zamiast tego należy uruchamiać
`predict()` dla każdej klatki. Informacje o źródłach, streamingu i obsłudze
wyników znajdują się w sekcji [predykcja](/docs/predict).

## Trenowanie

Żadna rodzina w tej warstwie nie jest trenowana wewnątrz LibreYOLO. `train()`
zgłasza błąd. Należy dostroić model w projekcie nadrzędnym i wczytać wynikowe
wagi. Słownik przekazywany do `set_classes()` jest jedynym ustawieniem, które
zmienia obiekty wykrywane przez wczytany model.

## Walidacja

Dla tej warstwy nie ma walidatora, a `val()` zgłasza błąd. Walidacja detekcji z
otwartym słownikiem wymaga specjalnego walidatora, ponieważ standardowy
walidator detekcji przekazuje tensory obrazów bezpośrednio do modelu, a te
rodziny wymagają równolegle zbudowanych wejść uwarunkowanych tekstem.

## Eksport

Eksport nie wchodzi w zakres tej warstwy, a `export()` zgłasza błąd. Modele te
działają przez `predict()` w środowisku PyTorch.


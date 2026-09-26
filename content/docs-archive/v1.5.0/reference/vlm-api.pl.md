---
title: API wizyjno-językowe
seo_title: 'API LibreVLM: aliasy, set_classes i chat'
description: >-
  Fabryka LibreVLM, wszystkie aliasy modeli, trwały słownik set_classes,
  set_task, swobodny interfejs chat oraz przyczyna używania pola zastępczego
  pewności.
lead: >-
  LibreVLM wczytuje generatywny model wizyjno-językowy i steruje nim jak
  detektorem obiektów. Lista klas jest podpowiedzią, a nie stałą głowicą, a
  model zwraca ten sam Results co każda inna rodzina.
keywords:
  - LibreVLM
  - detekcja modelem wizyjno-językowym
  - Qwen3-VL
  - LFM2-VL
  - InternVL3
  - SmolVLM2
  - Florence-2
  - LibreYOLO chat
last_verified: 1.5.0
verification: >-
  Aliasy odczytano z libreyolo/models/vlm/__init__.py; repozytoria, rozmiary i
  listy zadań z modułów rodzin w libreyolo/models/vlm/ oraz
  libreyolo/models/sensenova/model.py; reguły wywołań i wyjątki z
  libreyolo/models/vlm/base.py, wszystko w wersji 1.5.0.
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[vlm]'
  usage:
    - label: Detekcja z otwartym słownikiem
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("lfm2-vl-450m")
        model.set_classes(["person", "skateboard"])

        result = model.predict(SAMPLE_IMAGE)
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            print(result.names[int(cls)], box.tolist())
    - label: Zadawanie swobodnego pytania
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("lfm2-vl-450m")
        print(model.chat(SAMPLE_IMAGE, "How many people are in this image?"))
source_hash: 57ddac08bc4d4e05
---

## Instalacja

Ten poziom wymaga dodatku `vlm`.

<code-tabs name="install" />

## Fabryka

```python
LibreVLM(model: str = "qwen3-vl-4b", **kwargs) -> LibreVLMModel
```

`model` jest aliasem, a nie ścieżką. `**kwargs` trafia do konstruktora rodziny,
który przyjmuje `device`, `names` (początkowy słownik, równoważny wywołaniu
`set_classes` po wczytaniu), `prompt` (nadpisanie podpowiedzi detekcji) oraz
`max_new_tokens`. Nieznany alias zgłasza `ValueError` z listą wszystkich
aliasów.

<code-tabs name="usage" />

## Aliasy

| Rodzina | Aliasy | Rozmiary | Wagi |
|---|---|---|---|
| Qwen3-VL | `qwen3-vl`, `qwen3-vl-2b`, `qwen3-vl-4b`, `qwen3-vl-8b` | `2b`, `4b`, `8b` | `Qwen/Qwen3-VL-2B-Instruct`, `-4B-`, `-8B-` |
| LFM2-VL | `lfm2-vl`, `lfm2-vl-450m`, `lfm2-vl-1.6b` | `450m`, `1.6b` | `LiquidAI/LFM2.5-VL-450M`, `-1.6B` |
| InternVL3 | `internvl3`, `internvl3-1b`, `internvl3-2b`, `internvl3-8b` | `1b`, `2b`, `8b` | `OpenGVLab/InternVL3-1B-hf`, `-2B-hf`, `-8B-hf` |
| SmolVLM2 | `smolvlm2`, `smolvlm2-2.2b`, `smolvlm2-500m` | `2.2b`, `500m` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct`, `SmolVLM2-500M-Video-Instruct` |
| Florence-2 | `florence-2`, `florence2`, `florence-2-base`, `florence-2-large` | `base`, `large` | `florence-community/Florence-2-base`, `-large` |
| Kosmos-2 | `kosmos-2`, `kosmos2` | `224` | `microsoft/kosmos-2-patch14-224` |
| LocateAnything | `locate-anything`, `locateanything`, `locate-anything-3b`, `locateanything-3b` | `3b` | `nvidia/LocateAnything-3B` |
| SenseNova-Vision | `sensenova-vision`, `sensenova-vision-7b`, `sensenovavision` | `7b` | `LibreYOLO/SenseNovaVision7b` |
| LibreMODUS | `libremodus`, `libremodus-14b-a7b`, `modus`, `modus-14b-a7b` | `14b-a7b` | Przypięta migawka ze źródła nadrzędnego |

Domyślny alias to `qwen3-vl-4b`. Rozmiary domyślnego aliasu każdej rodziny są
wymienione jako pierwsze: `qwen3-vl` rozwiązuje się do `4b`, `lfm2-vl` do
`450m`, `internvl3` do `2b`, `smolvlm2` do `2.2b`, a `florence-2` do `base`.

`LibreVLM`, `LibreLFM2VL`, `LibreQwen3VL`, `LibreSmolVLM2`, `LibreInternVL3`,
`LibreFlorence2`, `LibreKosmos2`, `LibreLocateAnything` i `LibreMODUS`
(zapisywane również jako `LibreModus`) są eksportowane na poziomie pakietu.

## Zadania

Większość rodzin obsługuje tylko `detect`. Dwie obsługują więcej zadań:

| Rodzina | Obsługiwane zadania |
|---|---|
| LocateAnything | `detect`, `point` |
| SenseNova-Vision | `detect`, `segment`, `panoptic`, `pose`, `point`, `depth`, `ocr` |

Ponieważ zadanie jest sterowane podpowiedzią, a nie zapisane w checkpoincie,
można je przełączyć we wczytanym modelu:

```python
model.set_task(task: str) -> LibreVLMModel
```

Zadanie jest walidowane względem listy obsługiwanej przez rodzinę, pozostaje
ustawione w kolejnych wywołaniach `predict()` i `track()`, a model jest zwracany,
aby wywołania można było łączyć w łańcuch.

## set_classes

```python
model.set_classes(classes: list[str]) -> LibreVLMModel
```

Ustawia otwarty słownik. Działają dowolne słowa, ponieważ model otrzymuje je
jako podpowiedź, a nie ograniczenie stałej głowicy. Lista musi być niepusta,
a jej wpisy muszą być unikalne przy porównaniu bez uwzględniania wielkości
liter. Przekazanie samego ciągu zgłasza `TypeError`, ponieważ zostałby rozbity
na jednoznakowe klasy. Słownik jest trwały: wystarczy ustawić go raz po
wczytaniu, a pozostaje aktywny do kolejnego ustawienia.

## chat

```python
model.chat(image, prompt, max_new_tokens=None, color_format="auto") -> str
```

Surowe generowanie multimodalne: na wejściu obraz i podpowiedź, na wyjściu
dokładnie zdekodowany tekst. Jest to swobodny interfejs pod wygodną otoczką
detekcji, przeznaczony do pytań otwartych, liczenia lub formatu wyjścia, którego
otoczka detekcji nie obsługuje. `max_new_tokens` przechodzi na wartość
`MAX_NEW_TOKENS` rodziny, która w klasie bazowej wynosi 1024. Dekodowanie jest
zachłanne i stosuje łagodną karę za powtórzenia.

## Pewność

Wygenerowane wyjście nie ma skalibrowanej pewności dla poszczególnych ramek.
Ta wersja przypisuje stałe pole zastępcze, aby działały `predict`, rysowanie
i `track`, przez co filtrowanie `conf=` oraz mAP mają jedynie pozorny, a nie
rzeczywisty sens. Z tego samego powodu `val()` zgłasza błąd: COCO mAP oparte na
zastępczych wynikach wprowadzałoby w błąd.

## Predict i track

Obowiązuje standardowy interfejs predict, a `track()` działa, więc detektor VLM
można wstawić do tego samego pipeline'u co każdą inną rodzinę. Dwie polityki
klasowe różnią się od detektora konwolucyjnego: augmentacja podczas testu jest
wyłączona, ponieważ augmentacja wieloskalowa nie ma znaczenia dla generatora
o stałej rozdzielczości, a predykcja batchowa jest wyłączona, ponieważ
generowanie jest autoregresyjne, a przetwarzanie wstępne zwraca kodowanie tekstu
i obrazu zamiast tensora obrazów, które można ułożyć w stos.

## Nieobsługiwane funkcje

`train()`, `val()` i `export()` zgłaszają `NotImplementedError`. Dostrajanie
należy wykonać w projekcie nadrzędnym, a następnie wczytać wynikowe wagi.

## Kod zdalny

Każda dostarczana rodzina jest wczytywana przez natywną klasę modelu, więc
LibreYOLO domyślnie nie wykonuje kodu z zewnętrznego repozytorium. Rodzina,
która rzeczywiście go potrzebuje, musi jawnie wyrazić zgodę i przypiąć rewizję
migawki. Jedyną taką rodziną jest LocateAnything, przypięta do commita
`c32291ca5e996f5a7a485845b4f57a233936bba0`.

LibreMODUS jest jawnym wyjątkiem od schematu checkpointu: jego alias rozwiązuje
się do katalogu przypiętych plików ze źródła nadrzędnego zamiast pliku `.pt`
LibreYOLO, a LibreYOLO nie dodaje do niego metadanych v1.0 ani nie publikuje go
ponownie.

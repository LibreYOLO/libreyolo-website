---
title: API segmentacji sterowanej podpowiedziami
seo_title: 'API LibreSAM: podpowiedzi, aliasy i sygnatury'
description: >-
  Fabryka LibreSAM, jej aliasy rozmiarów, typy podpowiedzi punktowych, ramkowych
  i tekstowych, cykl encode-once z set_image oraz funkcje nieobsługiwane przez
  ten poziom.
lead: >-
  LibreSAM jest fabryką segmentacji sterowanej podpowiedziami. Przebieg w przód
  wymaga podpowiedzi dla obrazu dostarczonej podczas wywołania, dlatego ten
  poziom ma własny interfejs predict zamiast korzystać z modułu inferencji bez
  podpowiedzi.
keywords:
  - LibreSAM
  - segmentacja sterowana podpowiedziami
  - SAM podpowiedź punktowa
  - SAM podpowiedź ramka
  - set_image
  - segmentacja wszystkiego
  - dodatek sam LibreYOLO
last_verified: 1.5.0
verification: >-
  Aliasy fabryki, rozmiary i repozytoria odczytano z
  libreyolo/models/sam/model.py, sam2.py, edgetam.py, sam3.py,
  libreyolo/models/mobilesam/model.py i libreyolo/models/picosam3/model.py.
  Kontrakt podpowiedzi i wartości domyślne odczytano z
  libreyolo/models/sam/base.py. Założenia projektowe pochodzą z
  docs/adr/0007-libresam-contract.md, wszystko w wersji 1.5.0.
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[sam]'
  usage:
    - label: Podpowiedzi punktowe i ramkowe
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        r = model.predict(SAMPLE_IMAGE, points=[900, 370], labels=[1])
        print(r.masks.xy)
        print(r.boxes.xyxy)

        r = model.predict(SAMPLE_IMAGE, bboxes=[100, 100, 200, 200])
        print(len(r))
    - label: Jedno kodowanie i wiele podpowiedzi
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")
        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[500, 375], labels=[1])
        b = model.predict(bboxes=[100, 100, 200, 200])
        print(len(a), len(b))

        model.reset_image()
source_hash: 18e8206c10ce17fd
---

## Instalacja

Ten poziom wymaga dodatku `sam`.

<code-tabs name="install" />

## Fabryka

```python
LibreSAM(model: str = "base", **kwargs) -> LibreSAMModel
```

`model` jest aliasem rozmiaru, a nie ścieżką. `**kwargs` trafia do konstruktora
rodziny, który przyjmuje `device` i `multimask`. Nieznany alias zgłasza
`ValueError`, a komunikat wymienia wszystkie znane aliasy.

<code-tabs name="usage" />

## Aliasy

| Rodzina | Aliasy | Rozmiary | Wagi |
|---|---|---|---|
| SAM-1 | `base`, `large`, `huge`, `b`, `l`, `h`, `sam-base`, `sam-large`, `sam-huge`, `sam_b`, `sam_l`, `sam_h` | `base`, `large`, `huge` | `facebook/sam-vit-base`, `-large`, `-huge` |
| SAM-2 | `sam2-tiny`, `sam2-small`, `sam2-base-plus`, `sam2-baseplus`, `sam2-large` oraz krótkie formy `sam2-t`, `sam2-s`, `sam2-bp`, `sam2-l`, `sam2_t`, `sam2_s`, `sam2_bp`, `sam2_l` | `tiny`, `small`, `base-plus`, `large` | `LibreYOLO/LibreSAM2tiny`, `-small`, `-base-plus`, `-large` |
| EdgeTAM | `edgetam`, `edge-tam`, `edgetam-edge` | `edge` | `LibreYOLO/LibreEdgeTAM` |
| SAM 3 | `sam3`, `sam-3`, `sam3-large` | `large` | `facebook/sam3` |
| MobileSAM | `mobilesam`, `mobilesam-tiny`, `mobilesam_t`, `mobile-sam`, `mobile-sam-tiny` | `tiny` | `LibreYOLO/LibreMobileSAM` |
| PicoSAM3 | `picosam3`, `picosam3-pico`, `picosam3_pico`, `pico-sam3` | `pico` | `LibreYOLO/LibrePicoSAM3` |

Wartością domyślną jest `base`. SAM-1, SAM-2, EdgeTAM i MobileSAM działają na
nominalnym obszarze 1024 pikseli, SAM 3 na 1008, a PicoSAM3 na 96.

Dostęp do wag SAM 3 jest ograniczony. Są pobierane z `facebook/sam3` na
niestandardowej licencji SAM firmy Meta, która nie jest ani MIT, ani
Apache-2.0, i nie są rozpowszechniane przez LibreYOLO. Przed wczytaniem należy
zaakceptować warunki na stronie repozytorium i uwierzytelnić się w Hugging Face.
Loader najpierw zapisuje powiadomienie.

Klasy rodzin również są eksportowane, więc `LibreSAM1`, `LibreSAM2`,
`LibreSAM3`, `LibreEdgeTAM`, `LibreMobileSAM` i `LibrePicoSAM3` można tworzyć
bezpośrednio z `size=`.

## predict

```python
model.predict(
    source=None,
    *,
    points=None,
    bboxes=None,
    labels=None,
    masks=None,
    text=None,
    conf=None,
    multimask=None,
    max_det=300,
    device=None,
    color_format="auto",
    points_per_side=None,
) -> Results
```

| Argument | Wartość domyślna | Znaczenie |
|---|---|---|
| `source` | `None` | Obraz do segmentacji; `None` ponownie używa obrazu zapisanego w pamięci podręcznej przez `set_image()` |
| `points` | `None` | Podpowiedź punktowa we współrzędnych pikseli |
| `bboxes` | `None` | Podpowiedź ramkowa jako `[x1, y1, x2, y2]` albo ich lista, aby uzyskać po jednej masce na ramkę |
| `labels` | `None` | Etykiety punktów, `1` dodatnia i `0` ujemna, o kształcie zgodnym z `points`; w razie pominięcia wszystkie są dodatnie |
| `masks` | `None` | Zarezerwowane; przekazanie wartości zgłasza `NotImplementedError` |
| `text` | `None` | Podpowiedź koncepcji; tylko SAM 3 |
| `conf` | `None` | Dolna granica przewidywanego IoU maski |
| `multimask` | `None` | Zwracanie wszystkich masek niejednoznaczności dla każdej podpowiedzi; domyślnie zgodnie z ustawieniem konstruktora |
| `max_det` | `300` | Limit zwracanych masek |
| `device` | `None` | Przeniesienie modelu dla tego i kolejnych wywołań, z unieważnieniem embeddingów w pamięci podręcznej |
| `color_format` | `"auto"` | Wskazówka formatu kolorów dla tablic w pamięci |
| `points_per_side` | `None` | Gęstość siatki dla segmentacji wszystkiego; domyślnie 32 |

Zwracany jest zwykły `Results` zawierający `masks` oraz ciasne `boxes`
wyprowadzone z tych masek, z klasą `0` o nazwie `"object"`.

## Kształty podpowiedzi

`points` przyjmuje zagnieżdżone formy `[x, y]` dla jednego obiektu,
`[[x, y], ...]` dla N obiektów oraz `[[[x, y], ...], ...]` dla punktów
pogrupowanych według obiektu. Tablice numpy działają wszędzie tam, gdzie lista.
Współrzędne są zwykłymi pikselami obrazu źródłowego.

Pominięcie wszystkich podpowiedzi przestrzennych uruchamia segmentację
wszystkiego, czyli automatyczny generator masek na siatce z progiem
przewidywanego IoU i usuwaniem duplikatów według IoU ramek. Domyślne
`points_per_side` równe 32 wykonuje około 1024 przebiegów dekodera, co jest
powolne na CPU. Do zastosowań interaktywnych należy obniżyć tę wartość.
Generator pomija filtrowanie według wskaźnika stabilności, wielokrotne
przycinanie i usuwanie duplikatów według IoU masek, więc jest przybliżeniem
ścieżki z podpowiedziami, a nie jej dokładnym odpowiednikiem.

## Pewność

`conf` filtruje według przewidywanego IoU maski, czyli wyniku jakości maski,
a nie pewności detekcji. `None` zachowuje każdą maskę w ścieżce z podpowiedzią
i stosuje próg siatki rodziny w segmentacji wszystkiego. `0.0` wyłącza
filtrowanie w obu trybach.

W ścieżce tekstowej SAM 3 `conf` jest zamiast tego wynikiem detekcji Promptable
Concept Segmentation. `None` oznacza tam standardowy próg 0.3, a `0.0` zachowuje
wszystkich kandydatów.

## Podpowiedzi tekstowe

`text=` obsługuje tylko SAM 3. Każda rodzina podpowiedzi przestrzennych zgłasza
dla niego `NotImplementedError`. Tekst wzajemnie wyklucza się z punktami
i ramkami. Zwrócone `names` mapuje klasę `0` na żądaną koncepcję. Wywołanie
tekstowe z `source=None` ponownie koduje obraz z pamięci podręcznej, ponieważ
tracker i enkoder koncepcji nie korzystają ze wspólnej pamięci podręcznej.

Słowo kluczowe `exemplars=` jest zarezerwowane dla przyszłego rozszerzenia
z przykładami obrazów i nie jest zaimplementowane.

## Cykl jednokrotnego kodowania

```python
model.set_image(source, color_format="auto") -> LibreSAMModel
model.reset_image() -> LibreSAMModel
```

`set_image` wykonuje raz kosztowny enkoder obrazu i zapisuje embeddingi
w pamięci podręcznej, dzięki czemu każde późniejsze `predict()` z `source=None`
jest tanie. Obie metody zwracają model, więc wywołania można łączyć w łańcuch.
Przekazanie `device=` do `predict` przenosi model i unieważnia pamięć podręczną.

## PicoSAM3

PicoSAM3 przyjmuje tylko `bboxes=`. Podpowiedzi punktowe, tekstowe, maskowe,
multimask i segmentacja wszystkiego zgłaszają błąd. Ramka jest powiększana
o 10 procent i przetwarzana przez sieć ROI o rozmiarze 96 pikseli. PicoSAM3
jest jedyną rodziną na tym poziomie, która obsługuje eksport, wyłącznie do ONNX.

## Nieobsługiwane funkcje

`train()`, `val()` i `track()` zgłaszają `NotImplementedError` w każdej rodzinie
tego poziomu. Maski sterowane podpowiedziami nie mają stałego zbioru klas,
względem którego można je oceniać, więc mAP nie ma tu znaczenia. `export()`
zgłasza błąd dla SAM-1, SAM-2, SAM 3, EdgeTAM i MobileSAM.

Ścieżki wideo i pamięci dla SAM-2, SAM 3 i EdgeTAM wykraczają poza zakres tej
wersji, podobnie jak przykłady obrazów SAM 3 i podpowiedzi maskowe.

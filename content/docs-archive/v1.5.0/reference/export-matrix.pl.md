---
title: Pełna macierz eksportu
seo_title: Macierz obsługi eksportu LibreYOLO i jej reguły
description: >-
  Jak LibreYOLO określa możliwość eksportu połączenia rodziny, zadania i
  formatu: dwanaście formatów, trzy poziomy, reguły zachowania awaryjnego i
  progi zgodności.
lead: >-
  Obsługa eksportu jest wyszukiwana według trójki (rodzina, zadanie, format). Na
  tej stronie opisano kształt tej macierzy, reguły wypełniające komórki bez
  jawnego wpisu oraz sposób sprawdzania wybranej kombinacji.
keywords:
  - obsługa eksportu LibreYOLO
  - macierz eksportu
  - onnx tensorrt openvino tflite
  - polecenie libreyolo formats
  - próg zgodności eksportu
  - NotImplementedError eksport
last_verified: 1.5.0
verification: >-
  Formaty, poziomy, kolejność reguł awaryjnych, blokady zadań i rodzin oraz
  blokady NCNN odczytano z libreyolo/export/support.py; aliasy i wspólne
  argumenty z libreyolo/export/exporter.py; definicje poziomów z
  docs/adr/0011-export-support-tiers.md; progi zgodności z
  docs/export_support.md, wszystko w wersji 1.5.0. Nie przepisano tu komórek
  poszczególnych kombinacji. Należy odpytać je za pomocą poniższego fragmentu.
snippets:
  usage:
    - label: Odpytywanie macierzy bez modelu
      language: python
      code: |
        from libreyolo.export.support import (
            EXPORT_FORMATS,
            get_support,
            validated_alternatives,
        )

        print(EXPORT_FORMATS)

        entry = get_support("yolo9", "detect", "onnx")
        print(entry.tier, entry.since)
        print(entry.constraint)

        print(validated_alternatives("yolo9", "detect"))
    - label: CLI
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
        libreyolo formats --family yolo9 --task detect --json
  export:
    - label: Eksport i odczyt przyczyny odrzucenia
      language: python
      code: >
        from libreyolo import LibreYOLO

        from libreyolo.export.support import get_support


        model = LibreYOLO("LibreYOLO9t.pt")

        print(model.export(format="onnx"))


        # Sprawdź przed wywołaniem: zablokowana kombinacja zgłasza błąd w
        kontroli wstępnej,

        # a komunikat zawiera tę przyczynę.

        blocked = get_support("domedetr", "detect", "onnx")

        print(blocked.tier)

        print(blocked.reason)
source_hash: 83de3289634888c6
---

## Kształt macierzy

Kluczem macierzy jest `(family, task, format)`. Klucze rodzin są kanonicznymi
nazwami z rejestru modeli, klucze zadań pochodzą z `libreyolo.tasks.TASKS`,
a formatów jest dwanaście:

`onnx`, `torchscript`, `executorch`, `tensorrt`, `openvino`, `paddle`, `mnn`,
`rknn`, `ncnn`, `tflite`, `coreml`, `coreai`.

`model.export(format=...)` przyjmuje dodatkowo dwa aliasy: `engine` dla
`tensorrt` oraz `litert` dla `tflite`, czyli obecnej nazwy TensorFlow Lite.
Format i sufiks `.tflite` pozostają bez zmian.

<code-tabs name="usage" />

Ponieważ komórka jest funkcją trzech kluczy, pełna siatka jest duża i zmienia
się w każdym wydaniu. Jest generowana, a nie pisana ręcznie, i znajduje się
w `docs/export_support.md` w repozytorium biblioteki. Zamiast czytać kopię,
należy odpytać macierz z Pythona lub CLI.

## Trzy poziomy

| Poziom | Znaczenie |
|---|---|
| `validated` | Zgodność numeryczna jest objęta CI lub udokumentowanym przebiegiem nocnym |
| `available` | Konwersja jest zaimplementowana, ale nie zapisano dowodów zgodności numerycznej w środowisku uruchomieniowym |
| `blocked` | Kontrola wstępna zgłasza `NotImplementedError` z przyczyną przed śledzeniem |

Kombinacje validated i available przechodzą bez potwierdzenia ani ogólnego
ostrzeżenia. Zapisane dowody i ograniczenia pozostają widoczne w wygenerowanej
dokumentacji. Zablokowana kombinacja kończy się niepowodzeniem przed
sprawdzaniem zależności, wczytywaniem kalibracji, śledzeniem lub utworzeniem
artefaktu.

Dodanie wpisu validated wymaga testu zgodności i pola `since`.

`SupportEntry` zawiera cztery pola: `tier`, ciąg `reason`, wydanie `since`
i ciąg `constraint`. Podczas integracji znaczenie ma przede wszystkim
ograniczenie. Znacznik wyboru obowiązuje tylko w określonych w nim warunkach,
zwykle przy stałym obszarze wejściowym, batchu 1, FP32 i nazwanej wersji
środowiska uruchomieniowego.

## Sposób wyboru komórki

`get_support(family, task, fmt)` rozstrzyga w poniższej kolejności. Obowiązuje
pierwsza pasująca reguła.

1. Nieznane zadanie lub format spoza dwunastu zwraca `blocked`.
2. Jawny wpis `(family, task, format)` jest zwracany w zapisanej postaci.
3. Blokada całej rodziny zwraca `blocked` z przyczyną tej rodziny.
4. Blokada całego zadania zwraca `blocked` z przyczyną tego zadania.
5. Dla `ncnn` rodzina z listy blokad NCNN zwraca `blocked`.
6. `mnn` zwraca `blocked`: dla tej rodziny i zadania nie ma kontraktu środowiska uruchomieniowego.
7. `rknn` zwraca `blocked`. RKNN w tej wersji ogranicza się dokładnie do wariantów detekcji przetestowanych w symulatorze: YOLO9-t, YOLO9-E2E-t, YOLO-NAS-s i PicoDet-s na RK3588.
8. `tensorrt` i `openvino` zwracają `available`: ścieżka konwertera istnieje, ale nie zapisano zgodności w środowisku uruchomieniowym dla tej rodziny i zadania.
9. `tflite`, `paddle`, `coreai` i `coreml` zwracają `blocked`, każdy z własną przyczyną.
10. Wszystkie pozostałe przypadki zwracają `available`: konwersja jest zaimplementowana, ale nie zapisano zgodności numerycznej w środowisku uruchomieniowym.

Asymetria w krokach od 8 do 10 jest celowa. TensorRT i OpenVINO wykonują
ogólną konwersję z ONNX, więc warto podjąć próbę dla niewymienionej kombinacji.
TFLite, Paddle, Core AI i CoreML wymagają ścieżki właściwej dla rodziny, dlatego
niewymieniona kombinacja jest odrzuceniem, a nie zaproszeniem do próby.

## Zablokowane zadania

Te zadania są zablokowane dla każdej rodziny bez jawnego wpisu.

| Zadanie | Przyczyna |
|---|---|
| `ocr` | Dwie sieci z dynamicznym przycinaniem dla każdego regionu nie mieszczą się w kontrakcie eksportu pojedynczego grafu |
| `point` | Rodzina nie jest podłączona do wspólnego kontraktu mapy cieplnej punktów i dekodowania maksimów backendu |
| `semantic` | Rodzina nie jest podłączona do wspólnego kontraktu gęstych logitów i argmax backendu |
| `mesh` | Wyjścia grafu siatki ciała, metadane i kontrakt środowiska uruchomieniowego nie są zdefiniowane |
| `normal` | Rodzina nie jest podłączona do kontraktu gęstych normalnych jednostkowych na stałym obszarze i ponownej normalizacji backendu |
| `panoptic` | Eksport panoptyczny nie ma kontraktu środowiska uruchomieniowego backendu |
| `gaze` | Rodzina nie jest podłączona do wspólnego kontraktu logitów dwóch głowic i dekodowania wartości oczekiwanej backendu |

Jawny wpis nadpisuje te blokady. Dzięki temu może eksportować na przykład
podłączona rodzina semantic.

## Zablokowane rodziny

| Rodzina | Zakres blokady |
|---|---|
| `depth_anything3` | Każdy format; jej graf głębi nie należy do kontraktu wyeksportowanego środowiska uruchomieniowego |
| `domedetr` | Każdy format. PAQI ustala liczbę zapytań dla każdego obrazu, więc śledzony graf jest poprawny tylko dla obrazu użytego do śledzenia. Do eksportowalnego modelu DETR należy użyć D-FINE |
| `eomt` | Eksport instancji i panoptyczny, dla których brakuje parsowania w środowisku uruchomieniowym |
| `l2cs` | Wszystko poza ONNX, TorchScript, ExecuTorch, TensorRT i OpenVINO |
| `hrnet` | Wszystko poza ONNX, TorchScript, OpenVINO i TensorRT |
| `sam`, `sam2`, `sam3`, `edgetam`, `mobilesam` | Każdy format; eksport modelu sterowanego podpowiedziami wykracza poza kontrakt środowiska uruchomieniowego v1 |
| `grounding_dino`, `owlv2`, `omdet_turbo`, `ov_deim` | Każdy format; eksport środowiska uruchomieniowego z otwartym słownikiem wykracza poza zakres v1 |
| `florence2`, `kosmos2`, `lfm2vl`, `internvl3`, `qwen3vl`, `smolvlm2`, `locateanything` | Każdy format; eksport generatywnego VLM wykracza poza zakres v1 |

PicoSAM3 jest wyjątkiem na poziomie modeli sterowanych podpowiedziami:
eksportuje surową sieć ROI o rozmiarze 96 pikseli do ONNX.

## Zablokowane dla NCNN

Dekodery w stylu DETR wymagają operacji próbkowania, których NCNN nie
implementuje, dlatego poniższe rodziny są zablokowane dla `ncnn`, chyba że jawny
wpis stanowi inaczej: Deformable DETR, DETR, DINO-DETR, D-FINE, LW-DETR, DEIM,
DEIMv2, RT-DETR, RT-DETRv2, RT-DETRv4, RF-DETR i EC. Komunikat odrzucenia
wskazuje ONNX, OpenVINO, TorchScript i TensorRT jako alternatywy.

## Progi zgodności

Komórka validated oznacza, że wyeksportowany artefakt odtworzył model natywny
w poniższych granicach:

| Grupa zadań | Próg |
|---|---|
| Detekcja i OBB | IoU dopasowanych ramek powyżej 0.95, MAE wyników poniżej 0.01 |
| Segmentacja i zadania panoptyczne | IoU masek powyżej 0.95 |
| Poza | L2 punktów kluczowych poniżej 2 pikseli w natywnej rozdzielczości |
| Klasyfikacja | Podobieństwo cosinusowe logitów powyżej 0.999 i ta sama klasa top-1 |
| Głębia i przywracanie | PSNR powyżej 40 dB względem natywnego wyjścia |
| Normalne powierzchni | Średni błąd kątowy poniżej 0.1 stopnia |
| Punkt | Położenia maksimów zgodne z dokładnością do jednej komórki wyjścia |

Wiersze zapytań DETR są nieuporządkowanym zbiorem, dlatego zgodność rodziny
DETR wyrównuje wiersze zapytań jako zbiór, a nie pozycyjnie.

## Eksport

<code-tabs name="export" />

Zablokowana kombinacja zgłasza `NotImplementedError` podczas kontroli wstępnej,
a komunikat zawiera zapisaną przyczynę. `validated_alternatives(family, task)`
zwraca formaty zweryfikowane dla tej pary, co warto wyświetlić obok odrzucenia.

Argumenty wspólne dla wszystkich eksporterów wymieniono na stronie
[API modelu](/docs/reference/model-api). Argumenty właściwe dla formatów
opisano na stronach poszczególnych formatów.

## Interpretowanie ograniczenia

Komórka validated jest stwierdzeniem dotyczącym jednej zmierzonej konfiguracji,
a nie całego formatu. Ciąg ograniczenia, taki jak
`FP32, batch 1, fixed 520x520 input`, oznacza, że zgodność zapisano dla danego
kształtu i precyzji. Eksport z inną rozdzielczością lub rozmiarem batcha nadal
tworzy artefakt, ale nie jest konfiguracją, z której pochodzi wynik.

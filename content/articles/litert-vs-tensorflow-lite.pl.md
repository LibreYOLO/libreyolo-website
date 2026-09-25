---
title: "LiteRT a TensorFlow Lite: zmiana nazwy, nie nowy format"
description: "We wrześniu 2024 Google zmieniło nazwę TensorFlow Lite na LiteRT. To samo środowisko uruchomieniowe, te same pliki .tflite, nic się nie psuje. Oto powody tej zmiany, jej rzeczywisty zakres i sposób eksportu modelu LibreYOLO do tego formatu."
date: 2026-07-10
author: Xuban
tags: [LibreYOLO, litert, tensorflow-lite, tflite, export, edge-ai, tutorial]
faq:
  - q: "Czy TensorFlow Lite jest przestarzałe?"
    a: "Wycofywana jest nazwa, nie technologia. Środowisko uruchomieniowe działa dalej jako LiteRT z tymi samymi API, a istniejące modele i kod TFLite nadal działają. Nowe funkcje, takie jak API CompiledModel i akceleracja NPU, są udostępniane pod nazwą LiteRT."
  - q: "Czy trzeba ponownie wyeksportować modele do LiteRT?"
    a: "Nie. Plik .tflite wyeksportowany dla TensorFlow Lite jest modelem LiteRT. Zawartość pliku się nie zmieniła, więc nie trzeba go ponownie eksportować ani migrować."
---

**LiteRT to TensorFlow Lite pod nową nazwą. To nie jest nowe środowisko uruchomieniowe ani nowy format plików. Modele nadal są plikami `.tflite`, a każdy plik `.tflite`, który działał w TensorFlow Lite, działa bez zmian w LiteRT.**

Jeśli wyszukiwano „LiteRT vs TensorFlow Lite”, „czy TensorFlow Lite jest przestarzałe” albo „czym jest LiteRT”, odpowiedzią jest powyższy akapit. Dalsza część strony krótko wyjaśnia powód zmiany nazwy i sposób eksportowania modelu do tego formatu za pomocą LibreYOLO.

## Dlaczego Google zmieniło nazwę

TensorFlow Lite pojawiło się w 2017 roku jako sposób uruchamiania modeli TensorFlow na telefonach i płytkach wbudowanych. Z czasem zakres jego zastosowań znacznie się poszerzył: Google opracowało ścieżki konwersji z PyTorch, JAX i Keras, dlatego w 2024 roku duża część uruchamianych w nim modeli nigdy wcześniej nie miała do czynienia z TensorFlow.

Wtedy nazwa zaczęła wprowadzać w błąd. Użytkownicy PyTorch pomijali to narzędzie, bo jego nazwa sugerowała, że działa wyłącznie z TensorFlow. Dlatego we wrześniu 2024 Google [zmieniło jego nazwę na LiteRT](https://developers.googleblog.com/tensorflow-lite-is-now-litert/), skrót od „Lite Runtime”. To cała historia. Ten sam zespół, ten sam kod, nazwa niezależna od frameworka.

## Co się właściwie zmieniło

Niewiele, a żadna z tych zmian niczego nie psuje:

* Kod przeniesiono w nowe miejsce: [github.com/google-ai-edge/LiteRT](https://github.com/google-ai-edge/litert).
* Dokumentacja jest teraz dostępna pod adresem [ai.google.dev/edge/litert](https://ai.google.dev/edge/litert).
* Pakiet Pythona do uruchamiania modeli nazywa się teraz `ai-edge-litert`. Stary pakiet `tflite-runtime` jest nieaktualny; nowy udostępnia tę samą klasę `Interpreter`.
* Od zmiany nazwy nowe funkcje są udostępniane pod nazwą LiteRT: prostsze API `CompiledModel` oraz akceleracja GPU/NPU, którą Google uznało za [gotową do zastosowań produkcyjnych w styczniu 2026](https://developers.googleblog.com/litert-the-universal-framework-for-on-device-ai/).

Klasyczne API Interpreter nadal działa bez zmian. Istnieje jedno rzeczywiście nowe rozszerzenie, `.litertlm`, ale to format pakowania przeznaczony wyłącznie dla modeli LLM działających na urządzeniu; w przypadku modeli wizji komputerowej nie ma ono znaczenia. Jeśli więc jeden dokument mówi „TFLite”, a inny „LiteRT”, chodzi w nich o ten sam plik.

## Eksportowanie modelu LibreYOLO do LiteRT

LibreYOLO dodało ścieżkę eksportu TFLite/LiteRT w wersji v1.3.0. Wymagany jest Python 3.12+ i jeden dodatkowy pakiet:

```bash
pip install "libreyolo[tflite]"   # Python 3.12+
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9s.pt")  # auto-downloads on first run
model.export(format="tflite")        # writes a .tflite file
```

Można też użyć CLI:

```bash
libreyolo export model=LibreYOLO9s.pt format=tflite
```

Obecnie zweryfikowane ścieżki obejmują detekcję YOLO9 oraz detekcję, segmentację i estymację pozy w RF-DETR. Wszystkie nadal mają status eksperymentalny. Pełna macierz obsługiwanych funkcji znajduje się w [dokumentacji](/docs/v1.3.0).

Aby uruchomić wyeksportowany plik, zainstaluj na urządzeniu docelowym pakiet Google `ai-edge-litert`:

```bash
pip install ai-edge-litert
```

```python
import numpy as np
from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

image = np.zeros((1, 640, 640, 3), dtype=np.float32)  # NHWC, your preprocessed image here
interpreter.set_tensor(inp["index"], image)
interpreter.invoke()
predictions = interpreter.get_tensor(out["index"])
```

Zwróć uwagę, że wejście ma układ NHWC (kanały na końcu): podczas konwersji z ONNX układ jest transponowany, co jest normalne w tym formacie.

## Co zmieniliśmy w LibreYOLO

Niewiele, bo niewiele trzeba było zmieniać. W dokumentacji format jest teraz oznaczony jako „TFLite (LiteRT)”, aby można go było wyszukać pod obiema nazwami, a w API nazwa formatu nadal brzmi `format="tflite"`.

## FAQ

**Czy TensorFlow Lite jest przestarzałe?**
Wycofywana jest nazwa, nie technologia. Środowisko uruchomieniowe działa dalej jako LiteRT z tymi samymi API, a istniejące modele i kod TFLite nadal działają. Nowe funkcje, takie jak API CompiledModel i akceleracja NPU, są udostępniane pod nazwą LiteRT.

**Czy trzeba ponownie wyeksportować modele do LiteRT?**
Nie. Plik `.tflite` wyeksportowany dla TensorFlow Lite jest modelem LiteRT. Zawartość pliku się nie zmieniła, więc nie trzeba go ponownie eksportować ani migrować.

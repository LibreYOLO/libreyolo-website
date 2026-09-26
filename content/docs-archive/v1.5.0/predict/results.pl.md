---
title: Praca z wynikami
seo_title: Obiekt Results w LibreYOLO
description: >-
  Jeden obiekt Results na obraz z polem dla każdego typu danych: ramki, maski,
  punkty kluczowe, prawdopodobieństwa, głębia, segmentacja panoptyczna, OCR i
  inne. Rysowanie, zapisywanie i JSON.
lead: >-
  Każda predykcja zwraca obiekt Results dla każdego obrazu. Ma jedno nazwane
  pole dla każdego rodzaju danych, a wszystkie poza tymi tworzonymi przez model
  są puste. Te same pola występują dla wyeksportowanego artefaktu.
keywords:
  - obiekt Results YOLO Python
  - results.boxes xyxy
  - Results do JSON
  - zapis obrazu z adnotacjami
  - maski segmentacji Python
  - wyniki punktów kluczowych
  - wyniki mapy głębi
  - podsumowanie Results
  - ONNX ten sam Results
last_verified: 1.5.0
verification: >-
  Klasy danych wynikowych, pola, semantykę przenoszenia, summary(), to_json(),
  plot(), save() i cutout() odczytano z libreyolo/utils/results.py. Zachowanie
  adnotacji i zapisu na dysku pochodzi z InferenceRunner._save_annotated_image w
  libreyolo/models/base/inference.py oraz resolve_save_path w
  libreyolo/utils/general.py. Wybór według rozszerzenia pochodzi z LibreYOLO() w
  libreyolo/models/__init__.py.
snippets:
  basic:
    - label: Ramki
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(SAMPLE_IMAGE)


        print(result.orig_shape)   # (wysokość, szerokość) obrazu źródłowego

        print(result.path)         # ścieżka źródłowa, None dla wejścia w
        pamięci


        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            print(result.names[int(cls)], round(float(conf), 3), xyxy)
    - label: Współrzędne znormalizowane
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy[:1])    # piksele, x1 y1 x2 y2

        print(result.boxes.xywh[:1])    # piksele, środek x, środek y, w, h

        print(result.boxes.xyxyn[:1])   # ta sama ramka podzielona przez
        szerokość i wysokość

        print(result.boxes.xywhn[:1])
    - label: NumPy i urządzenia
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        # Każda z tych metod zwraca nowy Results; oryginał pozostaje bez zmian.
        as_numpy = result.numpy()
        on_cpu = result.cpu()

        print(type(as_numpy.boxes.xyxy).__name__)
        print(type(on_cpu.boxes.xyxy).__name__)
  json:
    - label: summary i to_json
      language: python
      code: |
        import json

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        rows = result.summary()
        print(json.dumps(rows[:2], indent=2))

        # Ta sama treść jako ciąg, z tymi samymi argumentami nazwanymi.
        print(result.to_json(normalize=True, decimals=3)[:200])
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt --json \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  saving:
    - label: Obrazy z adnotacjami
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # save=True rysuje dane wynikowe i zapisuje je w runs/detect/predict*.
        result = model(SAMPLE_IMAGE, save=True)
        print(result.saved_path)
  exported:
    - label: Instalacja dodatku eksportu
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: Ten sam Results z wyeksportowanego artefaktu
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        path = model.export(format="onnx")   # zwraca zapisaną ścieżkę

        # LibreYOLO() wybiera ścieżkę na podstawie rozszerzenia pliku.
        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)

        print(type(result).__name__, len(result.boxes))
source_hash: 548dbc9c7f5552ec
---

## Jeden obiekt i jedno pole na typ danych

Predykcja na jednym obrazie zwraca jeden obiekt `Results`. Zawiera on osiemnaście
pól danych wynikowych, a model wypełnia tylko te, które tworzy jego zadanie.
Każde pozostałe pole ma wartość `None`, więc odczytanie `result.masks` dla
detektora zwraca `None`, a nie błąd.

| Pole | Klasa | Kształt | Tworzone przez |
|---|---|---|---|
| `boxes` | `Boxes` | `(N, 4)` oraz wyniki i klasy | Detekcja i każde zadanie, które najpierw lokalizuje |
| `masks` | `Masks` | `(N, H, W)` | Segmentacja instancji |
| `keypoints` | `Keypoints` | `(N, K, 2)` albo `(N, K, 3)` | Estymacja pozy |
| `probs` | `Probs` | `(C,)` | Klasyfikacja |
| `obb` | `OBB` | `(N, 7)` albo `(N, 8)` | Ramki zorientowane |
| `gaze` | `Gaze` | `(N, 2)` kąt pochylenia i odchylenia w radianach | Estymacja kierunku wzroku |
| `points` | `Points` | `(N, 4)` jako x, y, klasa, pewność | Lokalizacja punktów |
| `semantic_mask` | `SemanticMask` | `(H, W)` identyfikatory klas | Segmentacja semantyczna |
| `panoptic` | `PanopticSegmentation` | `(H, W)` identyfikatory segmentów oraz `segments_info` | Segmentacja panoptyczna |
| `depth_map` | `DepthMap` | `(H, W)` liczby zmiennoprzecinkowe | Estymacja głębi |
| `normal_map` | `NormalMap` | `(H, W, 3)` wektory jednostkowe | Normalne powierzchni |
| `edges` | `EdgeMap` | `(H, W)` liczby zmiennoprzecinkowe w `[0, 1]` | Detekcja krawędzi |
| `restored` | `RestoredImage` | `(H, W, 3)` uint8 RGB | Rekonstrukcja i superrozdzielczość |
| `matte` | `Matte` | `(H, W)` liczby zmiennoprzecinkowe w `[0, 1]` | Wyznaczanie alfa i usuwanie tła |
| `ocr` | `OCRRegions` | `(N, 4, 2)` wielokąty oraz transkrypcje | Detekcja i rozpoznawanie tekstu |
| `embeddings` | `Embeddings` | `(N, D)` wiersze znormalizowane L2 | Zadanie `embed` |
| `identities` | `Identities` | N nazw i wyników | Zadanie `embed` z galerią |
| `meshes` | `Meshes` | Parametry ciała i opcjonalne wierzchołki | Rekonstrukcja siatki ciała |

Obok nich znajdują się pola obecne w każdym wyniku: `orig_shape` jako
`(height, width)`, `path` (ścieżka źródłowa albo `None` dla wejścia w pamięci),
`names` odwzorowujące identyfikator klasy na nazwę, `frame_idx` dla wideo i klatek
na żywo, `track_id` podczas śledzenia oraz `restore_scale`, czyli całkowity
współczynnik powiększenia wyniku rekonstrukcji.

`result.normals` jest aliasem `result.normal_map`.

`result.speed` istnieje w każdym wyniku, ale jest wypełniane tylko przez
[zespoły](/docs/predict/ensembling), gdzie jego klucze to `member_0`, `member_1`
i `fusion`, a wartości są podane w milisekundach. Dla pojedynczego modelu
pozostaje pustym słownikiem.

## Ramki

<code-tabs name="basic" />

`Boxes` przechowuje współrzędne i wyniki jako osobne tablice, a nie jeden spakowany tensor.

| Atrybut | Zawartość |
|---|---|
| `xyxy` | `(N, 4)` piksele bezwzględne, x1 y1 x2 y2 |
| `xywh` | `(N, 4)` piksele bezwzględne, środek x, środek y, szerokość, wysokość |
| `xyxyn`, `xywhn` | Te same wartości podzielone przez szerokość i wysokość obrazu |
| `conf` | `(N,)` pewność |
| `cls` | `(N,)` identyfikator klasy jako liczba zmiennoprzecinkowa |
| `id` | `(N,)` identyfikator śledzenia albo `None` |
| `is_track` | Czy `id` jest ustawione |
| `data` | Wszystko połączone: ramki, opcjonalne id, conf, cls |

`cls` jest tablicą zmiennoprzecinkową, dlatego używaj go jako `result.names[int(cls)]`.

`xyxyn` i `xywhn` wymagają `orig_shape`, które `Results` wypełnia automatycznie.

## Gęste dane wynikowe

Dane obejmujące cały obraz zachowują się inaczej niż dane poszczególnych instancji,
co ma znaczenie podczas wycinania.

`SemanticMask` zawiera identyfikatory klas `(H, W)` na oryginalnym płótnie, z
wartością `255` zarezerwowaną jako ignorowana i nigdy nieliczoną jako klasa.
`classes` wymienia obecne identyfikatory i ją pomija, a `class_mask(id)` zwraca
tablicę logiczną `(H, W)`.

`PanopticSegmentation` zawiera identyfikatory segmentów `(H, W)`, z `0` jako
identyfikatorem pustym, oraz listę słowników `segments_info` zawierających co
najmniej `id` i `category_id`. `segment_ids` wymienia obecne identyfikatory,
a `segment_mask(id)` wybiera jeden z nich.

`DepthMap` zawiera względną odwrotność głębi `(H, W)`: wyższa wartość oznacza
mniejszą odległość, a wartości nie są metrami. Udostępnia `min`, `max` i `mean`
dla wartości skończonych, a `normalized()` przeskalowuje je do `[0, 1]`.

`NormalMap` zawiera wektory jednostkowe `(H, W, 3)` w układzie kamery OpenCV,
gdzie `+x` wskazuje w prawo, `+y` w dół, a `+z` w głąb sceny. Powierzchnia
skierowana do kamery ma więc wektor `(0, 0, -1)`. `assert_normalized()` sprawdza,
czy każdy piksel jest skończony i ma długość jednostkową.

`EdgeMap` zawiera float32 `(H, W)` w `[0, 1]`. Ciągła mapa pozostaje bez progowania,
więc wartość graniczną wybiera się w `binary(threshold=0.5)`.

`Matte` zawiera float32 `(H, W)` w `[0, 1]`, gdzie `1` oznacza pełny pierwszy
plan. `array` zwraca ją przyciętą jako float32.

`RestoredImage` zawiera uint8 RGB `(H, W, 3)`, z `array` udostępniającym surową
tablicę ndarray i `save(path)` zapisującym obraz.

`Probs` zawiera jeden wektor prawdopodobieństw dla obrazu. `top1` i `top5` to
indeksy klas, a `top1conf` i `top5conf` odpowiadające im wyniki.

`Embeddings` zawiera wiersze `(N, D)` już znormalizowane L2, dlatego podobieństwo
cosinusowe jest iloczynem skalarnym. `similarity(other)` zwraca `(N, M)` względem
galerii albo `(N,)` względem pojedynczego wektora, a `verify(i, j, threshold=0.4)`
porównuje dwa wiersze.

`OCRRegions` zawiera wielokąty `(N, 4, 2)` w kolejności czytania, z narożnikami
w kolejności lewy górny, prawy górny, prawy dolny, lewy dolny. Transkrypcje są
w `texts`, wyniki rozpoznawania w `conf`, a wyniki detekcji w `det_conf`. Ponieważ
są to rzeczywiste obrócone wielokąty, nie wypełniają `boxes`. `ocr.xyxy` zwraca
wyrównane do osi obwiednie, gdy potrzebne są prostokąty.

## Wycinanie i przenoszenie

`result[i]` zwraca nowy obiekt `Results` zawierający jedną instancję. Dane
poszczególnych instancji są wycinane, a dane całego obrazu przechodzą bez zmian.
Dzięki temu wycięcie wyniku klasyfikacji nie skraca wektora prawdopodobieństwa
do jednej klasy, a wycięcie wyniku głębi nie psuje układu `(H, W)`.

`len(result)` liczy instancje: ramki, punkty, osadzenia, regiony OCR albo siatki.
Każde gęste dane całego obrazu liczą się jako `1`. Wynik bez zawartości ma długość `0`.

`to()`, `cpu()`, `cuda()` i `numpy()` zwracają nowy obiekt `Results` ze wszystkimi
wypełnionymi polami po konwersji. Nie modyfikują oryginału.

`update()` jest jedyną metodą modyfikującą obiekt w miejscu. Zastępuje nazwane
pola i zwraca ten sam obiekt.

## JSON

<code-tabs name="json" />

`summary()` zwraca listę zwykłych słowników, a `to_json()` przekazuje tę listę
do `json.dumps`. Obie metody przyjmują te same trzy argumenty: `normalize=False`
przełącza współrzędne na `[0, 1]`, `decimals=5` ustala zaokrąglenie, a
`embeddings=False` określa, czy uwzględnić wektory osadzeń.

Kształt wiersza zależy od danych. Wiersze detekcji zawierają `name`, `class`,
`confidence` i słownik `box`, a także `segments`, gdy obecne są maski, `obb` i
`corners` dla ramek zorientowanych, kąty `gaze` w radianach i stopniach, `track_id`
podczas śledzenia oraz parametry `mesh`, gdy obecne są siatki.

Gdy nie ma ramek, jeden rodzaj danych określa wiersze: OCR tworzy jeden wiersz
na region z jego `text`, punkty jeden wiersz na punkt, segmentacja panoptyczna
jeden wiersz na segment z `pixel_count` i `pixel_fraction`, semantyczna jeden
wiersz na obecną klasę, a klasyfikacja pięć najlepszych klas. Głębia, normalne,
krawędzie, rekonstrukcja i wyznaczanie alfa tworzą po jednym wierszu podsumowania
opisującym mapę zamiast jej pikseli.

Dwa rodzaje danych są celowo skracane. Wektor osadzenia jest domyślnie raportowany
tylko jako `embedding_dim`, ponieważ wiersz 512 liczb zmiennoprzecinkowych zajmuje
około 2 KB na twarz. Przekaż `embeddings=True`, aby uwzględnić wartości. Wierzchołki
siatki nigdy nie są uwzględniane, ponieważ oznaczałoby to dziesiątki tysięcy
współrzędnych na osobę. Geometrię można odczytać z `result.meshes.vertices`
albo zapisać przez `result.meshes.save_obj(path)`.

## Rysowanie i zapisywanie

<code-tabs name="saving" />

`predict(save=True)` wykonuje adnotację i zapis. Wybiera procedurę rysowania na
podstawie wypełnionego pola, więc wynik semantyczny jest zapisywany jako kolorowa
maska, wynik głębi jako wizualizacja głębi, wynik panoptyczny z segmentami, matte
jako plik PNG RGBA z przezroczystym tłem, a wynik detektora jako ramki z maskami
pod nimi. Zapisana ścieżka jest dołączana do wyniku jako `result.saved_path`.

`Results.plot()` ma węższy zakres, niż sugeruje nazwa. Jest zdefiniowane wyłącznie
dla map normalnych i krawędzi, a dla pozostałych danych zgłasza `NotImplementedError`.
Dla innych zadań używaj `save=True`.

`Results.save(path)` również ma wąski zakres. Zapisuje wynik matte jako wycięcie
PNG RGBA z przezroczystym tłem, a w innych przypadkach zgłasza `NotImplementedError`.
`Results.cutout()` zwraca tę samą tablicę RGBA bez zapisywania. Obie metody
potrzebują obrazu źródłowego pobranego z `result.path` albo przekazanego jako `image=`.

Dwa rodzaje danych mają własne metody zapisu: `result.restored.save(path)` dla
zrekonstruowanego obrazu oraz `result.meshes.save_obj(path, index=0)` dla siatki.

Informacje o lokalizacji plików oraz zachowaniu `output_path` i
`output_file_format` znajdziesz w sekcji [źródła predykcji](/docs/predict/sources).

## Wyeksportowane artefakty zwracają ten sam obiekt

<code-tabs name="exported" />

`LibreYOLO()` wybiera ścieżkę na podstawie rozszerzenia pliku, dlatego
wyeksportowany artefakt ładuje się za pomocą tego samego wywołania co punkt
kontrolny `.pt` i zwraca ten sam obiekt `Results`. Pliki `.onnx`, `.engine`,
`.pte` i `.mnn` są rozpoznawane według rozszerzenia, podobnie jak katalogi
OpenVINO, Paddle i ncnn oraz adres URL modelu Triton. Kod odczytujący
`result.boxes.xyxy` nie zmienia się po zastąpieniu modelu jego wyeksportowaną
wersją. Pełny zestaw formatów znajduje się w sekcji [eksport](/docs/export).

Użycie własnego API środowiska uruchomieniowego oznacza samodzielną odpowiedzialność
za przetwarzanie wstępne, końcowe i nazwy klas.


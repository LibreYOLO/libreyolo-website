---
title: Schemat checkpointu
seo_title: Schemat metadanych checkpointu LibreYOLO v1.0
description: >-
  Metadane zawarte w każdym checkpoincie .pt LibreYOLO: wymagane klucze, dodatki
  zależne od zadania, klucze środowisk uruchomieniowych eksportu, manifesty
  kwantyzacji i pola trenowania.
lead: >-
  Plik .pt LibreYOLO jest płaskim słownikiem zapisanym za pomocą torch.save.
  Klucz model zawiera słownik stanu, a pozostałe klucze najwyższego poziomu są
  metadanymi identyfikującymi checkpoint bez analizowania nazwy pliku ani
  rozpoznawania słownika stanu.
keywords:
  - schemat checkpointu LibreYOLO
  - schema_version 1.0
  - model_family
  - metadane checkpointu LibreYOLO
  - manifest kwantyzacji
  - wrap_libreyolo_checkpoint
last_verified: 1.5.0
verification: >-
  Odpowiada plikowi docs/checkpoint_schema.md w repozytorium libreyolo w wersji
  1.5.0 i została sprawdzona z libreyolo/utils/serialization.py oraz
  BaseModel.save.
snippets:
  usage:
    - label: Odczyt metadanych z checkpointu
      language: python
      code: >
        from libreyolo import LibreYOLO

        from libreyolo.utils.serialization import unwrap_libreyolo_checkpoint

        import torch


        # Pobierz checkpoint, a następnie zapisz go ponownie, aby utworzyć
        ścieżkę lokalną.

        LibreYOLO("LibreYOLO9t.pt").save("roundtrip.pt")


        loaded = torch.load("roundtrip.pt", map_location="cpu",
        weights_only=False)

        state_dict, metadata = unwrap_libreyolo_checkpoint(loaded)


        print(metadata["schema_version"], metadata["model_family"])

        print(metadata["size"], metadata["task"], metadata["nc"],
        metadata["imgsz"])

        print(len(state_dict), "tensors")
source_hash: ce760f1bed97bfd0
---

## Schemat v1.0

Każdy oficjalny checkpoint `.pt` LibreYOLO zawiera:

```python
{
    "model": state_dict,
    "schema_version": "1.0",
    "libreyolo_version": "0.x.y",
    "model_family": "yolo9",
    "size": "t",
    "task": "detect",
    "nc": 80,
    "names": {0: "cat", 1: "dog"},
    "imgsz": 640,
}
```

| Klucz | Typ | Znaczenie |
|---|---|---|
| `model` | state dict | Wagi modelu |
| `schema_version` | str | Wersja kontraktu metadanych; v1.0 używa ciągu `"1.0"` |
| `libreyolo_version` | str | Wersja, która utworzyła checkpoint |
| `model_family` | str | Zarejestrowana rodzina, na przykład `yolo9`, `rfdetr`, `dfine`, `ec` |
| `size` | str | Wariant w obrębie rodziny, na przykład `t`, `s`, `r18`, `atto` |
| `task` | str | Kanoniczna nazwa zadania |
| `nc` | int | Dodatnia liczba klas |
| `names` | dict | `dict[int, str]` z kluczami w zakresie `0..nc-1` |
| `imgsz` | int | Dodatnia rozdzielczość kwadratowego wejścia albo starsza wartość skalarna dla kontraktu prostokątnego |

`task` przyjmuje jedną z wartości `detect`, `segment`, `semantic`, `panoptic`,
`pose`, `classify`, `gaze`, `obb`, `point`, `depth`, `edge`, `normal`, `restore`,
`matte`, `ocr`, `embed` lub `mesh`.

Oficjalne checkpointy zapisują każdy klucz `names`. Czytniki mogą uzupełnić
brakujące klucze etykietami `class_i` dla starszych, rzadkich mapowań, ale
klucze spoza zakresu są nieprawidłowe.

Checkpointy prostokątne zachowują skalarną wartość `imgsz` dla starszych
czytników, ustawioną na `max(imgsz_h, imgsz_w)`, a dodatkowo zapisują
`imgsz_h` i `imgsz_w` z rzeczywistymi wymiarami. Czytnik rozumiejący pola
prostokątne musi preferować je względem wartości skalarnej. Rodziny o stałym
kontrakcie prostokątnym, takie jak HRNet pose, odrzucają niezgodne rozmiary
środowiska uruchomieniowego.

Schemat jest celowo płaski, a `model` celowo jest słownikiem stanu.

<code-tabs name="usage" />

## Dodatki dla pozy

Poza jest zwykle jednoklasowa, z `nc: 1` i `person`, ale głowica pozy YOLO-NAS
obsługuje także wieloklasową pozę z jednym wspólnym szkieletem punktów
kluczowych. W takim przypadku `nc` i `names` opisują klasy tak jak w detekcji.
Eksporty pozy dla środowiska uruchomieniowego emitują `scores` o kształcie
`[batch, anchors, nc]`.

| Klucz | Znaczenie |
|---|---|
| `num_keypoints` | Dodatnia liczba punktów kluczowych używana przez głowicę pozy |
| `keypoint_dim` | `2` dla etykiet `x,y` lub `3` dla etykiet `x,y,visibility`; wyniki modelu zawsze udostępniają `x,y,visibility` |
| `oks_sigmas` | Opcjonalne wartości sigma OKS dla poszczególnych punktów kluczowych; w razie braku używana jest wartość domyślna zadania dla `num_keypoints` |
| `num_keypoints_per_class` | Opcjonalna liczba punktów kluczowych dla poszczególnych klas w głowicach w stylu GroupPose, których tensor punktów kluczowych jest dopełniany według klasy; `0` dla klas bez punktów kluczowych |

## Dodatki dla siatki

Checkpointy siatki używają `task: "mesh"`, `nc: 1` i
`names: {0: "person"}`. Układy parametrów różnią się między modelami ciała,
dlatego wymiary są zapisywane, a nie przyjmowane z góry.

| Klucz | Znaczenie |
|---|---|
| `body_model` | Parametryzacja, na przykład `mhr`; pole wymagane, używane do interpretacji wszystkich poniższych pól |
| `num_betas` | Liczba współczynników tożsamości i kształtu; 45 dla MHR |
| `num_body_pose` | Szerokość bloku parametrów pozy ciała; 130 dla MHR. Jest to płaski wektor, a nie jedna trójka na staw, ponieważ stawy szkieletu mają różne stopnie swobody |
| `num_vertices` | Liczba wierzchołków emitowanych przez dekoder; 18439 dla MHR |
| `num_joints` | Liczba stawów emitowanych przez dekoder; 127 dla MHR |
| `rotation_format` | Sposób kodowania obrotów, na przykład `euler_zyx` dla MHR lub `axis_angle`. Nigdy nie jest wnioskowany z kształtu tensora, ponieważ wektor trójelementowy jest niejednoznaczny |

## Pola zastępcze zadań gęstych

Kilka zadań przewiduje gęste mapy zamiast klas, dlatego pola podobne do klas
istnieją tylko dla zgodności ze schematem.

| Zadanie | `nc` | `names` |
|---|---|---|
| `depth` | 1 | `{0: "depth"}` |
| `edge` | 1 | `{0: "edge"}` |
| `restore` | 1 | `{0: "image"}` |
| `ocr` | 1 | `{0: "text"}` |

Predykcje krawędzi są gęstymi mapami prawdopodobieństwa float32 w zakresie
`[0, 1]`.

Checkpointy przywracania mogą dodawać `degradation`, krótką etykietę
zniekształcenia, taką jak `deblur`, `denoise` lub `super-resolution`;
`dataset`, etykietę pochodzenia, taką jak `GoPro` lub `SIDD`; oraz `scale`,
dodatni całkowity współczynnik powiększenia wyjścia względem wejścia, na
przykład `4` dla modelu super-rozdzielczości x4. Brak pola lub wartość `1`
oznacza, że przywrócony obraz zachowuje rozdzielczość wejściową. Środowisko
uruchomieniowe wyznacza skalę również z rodziny i rozmiaru, więc `scale` jest
metadaną pochodzenia, a nie wymaganiem podczas wczytywania.

## Dodatki dla OCR

Rodzina `ppocr` dostarcza po jednym złożonym checkpoincie na poziom, którego
słownik stanu `model` zawiera dwa podmodele w przestrzeniach nazw kluczy
`det.*` i `rec.*`.

| Klucz | Znaczenie |
|---|---|
| `charset` | Pełny alfabet CTC w kolejności indeksów wyjścia: indeks 0 to pusty symbol CTC, potem słownik rozpoznawania, a następnie znak spacji. Loadery muszą odczytywać go z checkpointu, nigdy z pliku pomocniczego |
| `pipeline` | Wartości domyślne pipeline'u zapisane podczas konwersji: `det_limit_side_len`, `det_db_thresh`, `det_db_box_thresh`, `det_db_unclip_ratio`, `rec_image_shape`. Argumenty środowiska uruchomieniowego mogą nadpisywać je dla poszczególnych wywołań |
| `components` | Zarezerwowane dla opcjonalnych etapów pipeline'u, takich jak orientacja dokumentu, prostowanie i obrót wierszy tekstu. Puste w v1 |

## Metadane eksportu dla środowiska uruchomieniowego

Wyeksportowane artefakty stosują tę samą konwencję podwójnego zapisu wymiarów
prostokątnych: `imgsz_h` i `imgsz_w` są zapisywane obok starszej skalarnej
wartości `imgsz`, a czytnik, który nie rozumie pól prostokątnych, nie może po
cichu traktować wartości skalarnej jako kontraktu kwadratowego.

Obsługa prostokątnych danych w środowisku uruchomieniowym zależy od rodziny
i formatu. Eksporty rodziny YOLO9, HRNet, NAFNet i Real-ESRGAN mogą używać
niekwadratowych `imgsz_h` i `imgsz_w` w obsługiwanych formatach. Rodziny lub
formaty bez jawnej obsługi prostokątów odrzucają metadane zamiast przetwarzać
takie artefakty jako kwadratowe. Eksporty HRNet są stałymi głowicami FP32 dla
jednoelementowego batcha z wyciętym obszarem osoby. W32 przyjmuje 256x192,
a W48 przyjmuje 384x288. Detektor osoby nie jest osadzony w grafie.

Eksporty z osadzonym NMS mogą dodawać następujące płaskie klucze:

| Klucz | Znaczenie |
|---|---|
| `nms` | Wartość logiczna jako ciąg; `"true"` oznacza, że graf zawiera osadzone wyjście przetwarzania końcowego |
| `nms_conf` | Próg pewności zapisany w osadzonym wyjściu |
| `nms_iou` | Próg IoU zapisany w osadzonym wyjściu |
| `max_det` | Maksymalna liczba wierszy detekcji po NMS emitowana przez osadzone wyjście |
| `nms_raw_output` | Wartość logiczna jako ciąg; `"true"` oznacza, że graf udostępnia również pomocnicze surowe wyjście detektora |

Dla eksportów detekcji YOLO9 w formacie ONNX z `nms=true`, wyjście `0`
(o nazwie `output`) jest samodzielnym tensorem po NMS przy progach ustawionych
podczas eksportu. Gdy `nms_raw_output=true`, wyjście `1` (o nazwie `raw`) jest
zarezerwowane dla backendów LibreYOLO, aby mogły stosować natywne przycinanie
do oryginalnego obszaru roboczego oraz semantykę środowiska uruchomieniowego
`predict(conf=..., iou=..., max_det=...)`. Zewnętrzni użytkownicy powinni używać
pierwszego wyjścia.

Eksporty pozy mogą dodawać `num_keypoints`; `keypoint_dim`, przy czym surowe
eksporty w stylu GroupPose mogą używać większych wartości, takich jak `8`, gdy
tensor zawiera pola precyzji lub logitów klas; `num_keypoints_per_class` jako
listę zakodowaną w JSON, gdzie należy zachować pola klas bez punktów kluczowych,
ponieważ definiują schemat; oraz `pose_input`, gdzie `"person_crop"` oznacza,
że graf pobiera jeden już wycięty obszar i nie zawiera detektora. Eksporty HRNet
dla środowiska uruchomieniowego wymagają tej wartości.

Eksporty klasyfikacji mogą dodawać `crop_pct`, zmiennoprzecinkowy współczynnik
wycięcia ze środka, dla którego docelowy rozmiar przed przycięciem wynosi
`round(imgsz / crop_pct)` i który domyślnie wynosi `0.875` w razie braku, oraz
`interpolation`, `"bilinear"` lub `"bicubic"`, z domyślną wartością
`"bilinear"`.

Eksporty ExecuTorch zapisują płaskie metadane w wymaganym pliku pomocniczym
`<program>.pte.json`. Kontrakt v1 obejmuje CPU, FP32, batch 1 i stały obszar
wejściowy, a ponadto wymaga `executorch_version`, `executorch_delegate` równego
`"xnnpack"` i dodatniej wartości `executorch_delegate_partitions`. Loader
odrzuca plik pomocniczy deklarujący innego delegata, dynamiczne kształty lub
precyzję inną niż FP32.

Eksporty MNN zapisują płaskie metadane w wymaganym pliku pomocniczym
`<model>.mnn.json`. Kontrakt v1 obejmuje CPU, FP32, tylko detekcję i stały kształt
wejścia NCHW, a ponadto wymaga `mnn_version`, `mnn_backend` równego `"cpu"`,
uporządkowanych i niepustych `mnn_input_names` oraz `mnn_output_names`, pola
`mnn_input_shape` zawierającego cztery dodatnie liczby całkowite w kolejności
`[batch, channels, height, width]` oraz `mnn_batch` równego
`mnn_input_shape[0]`. Loader odrzuca dynamiczne metadane, precyzję inną niż
FP32, zadania inne niż detekcja, nieobsługiwane rodziny i niespójne kształty.

Pliki `.pte` i `.mnn` są artefaktami właściwymi dla backendu, a nie
checkpointami PyTorch.

## Checkpointy skwantyzowane

Skwantyzowany model dodaje jeden opcjonalny płaski klucz `quant`, zawierający
słownik manifestu z `schema`, `recipe`, `keep_high_precision`, `execution`,
pochodzeniem kalibracji, `module_count` i `state`. Manifesty FP8 mogą również
zawierać `fp8_tensorwise_weights`, dokładną listę nazw modułów `QuantLinear`,
których skala wag jest wspólna dla tensora, a nie osobna dla każdego kanału
wyjściowego. Loader, który napotka `quant`, odtwarza strukturę skwantyzowanego
modułu i politykę skalowania przed `load_state_dict`.

`state` rozróżnia dwie formy artefaktu.

`"prepared"`, wartość domyślna, zawiera główne wagi FP32 oraz bufory skali
`_q_*` i umożliwia trenowanie. Czytnik bez obsługi kwantyzacji może zignorować
klucz `quant` i wczytać główne wagi jako model zmiennoprzecinkowy.

`"finalized"` jest formą wdrożeniową zapisywaną przez `export(format="pt")`.
Główne wagi są usuwane, a każdy skwantyzowany moduł zawiera zamiast nich
spakowane wagi:

| Receptura | Spakowane tensory | Dekwantyzacja |
|---|---|---|
| int8 | `weight_packed` int8 o pierwotnym kształcie wag, `_q_w_scale` FP32 osobno dla każdego kanału | `weight_packed * scale` |
| fp8 | `weight_packed` float8_e4m3fn o pierwotnym kształcie, `_q_w_scale` FP32 z jednym wpisem na kanał wyjściowy | `weight_packed * scale` |
| w4a16, w4a8 | `weight_packed` uint8, dwa 4-bitowe kody w każdym bajcie, najpierw młodszy półbajt, kod `q + 8`; `_q_w_gscale` FP32 `[out, ngroups]`, grupa 128 wzdłuż in_features | Skala według grupy |
| int2 | Cztery 2-bitowe kody w każdym bajcie, kod `q + 2`, grupa 64 | Skala według grupy |
| nvfp4 | `weight_packed` uint8 `[out, ceil(in/16)*8]`, kod `sign<<3 \| E2M1 level`; `weight_block_scale` float8_e4m3fn `[out, ceil(in/16)]`; `_q_w_amax` FP32 na tensor | `block_scale * amax / (448 * 6)` |
| mxfp4 | Jak nvfp4, ale z blokami 32-elementowymi i dodatkowym `weight_block_exp` int8 `[out, ceil(in/32)]` | `2 ** exponent` |

Bufory zakresu aktywacji `_q_act_lo`, `_q_act_hi` i `_q_calibrated` są
zachowywane dla int8. Manifest zapisuje `remainder`, `"fp16"` lub `"fp32"`,
dla nieskwantyzowanych tensorów. Rozpakowanie odtwarza symulację bit po bicie,
więc sfinalizowana inferencja dokładnie odpowiada przygotowanej inferencji na
urządzeniu użytym do finalizacji. Ten układ jest stabilnym kontraktem dla
zewnętrznych eksporterów i środowisk uruchomieniowych.

## Checkpointy trenowania

Checkpointy trenera używają tego samego wymaganego rdzenia metadanych i mogą
dodawać płaskie pola trenowania oraz wznawiania:

```python
{
    "model": state_dict,
    "epoch": 42,
    "optimizer": optimizer_state_dict,
    "config": {},
    "loss": 1.23,
    "best_metric_key": "metrics/mAP50-95",
    "best_metric_value": 0.51,
    "best_epoch": 39,
    "is_ema_weights": True,
    "train_model": raw_state_dict,
    "ema": ema_state_dict,
    "ema_updates": 12345,
}
```

`is_ema_weights` określa, czy `model` najwyższego poziomu jest wygładzony przez
EMA. Gdy EMA jest włączone, `train_model`, `ema` i `ema_updates` zachowują stan
potrzebny do wznowienia. Opublikowane wagi do inferencji powinny być odchudzone
i nie powinny zawierać optymalizatora, epoki, konfiguracji, funkcji straty ani
stanu wznawiania EMA, chyba że są celowo rozpowszechniane jako checkpointy
trenowania.

Dla zgodności między wydaniami czytniki przyjmują starsze aliasy najlepszej
metryki: `best_mAP50_95`, `best_mAP50`, `best_metric` i `best_metric_name`.

## Zewnętrzne migawki

Schemat dotyczy plików `.pt` utworzonych przez LibreYOLO. Nie zmienia nazw ani
nie opakowuje wieloplikowych migawek ze źródeł nadrzędnych, używanych przez
oddzielne poziomy modeli.

Rozmiar LibreMODUS `14b-a7b` stanowi jawny wyjątek: alias jest rozwiązywany
przez `LibreVLM(...)` do katalogu przypiętych plików ze źródła nadrzędnego,
a LibreYOLO nie dodaje do niego metadanych v1.0 ani nie publikuje go ponownie
jako pliku `.pt`.

## Starsze i obce wagi

Nowe moduły zapisujące wykonują ścisłą walidację i muszą emitować metadane
v1.0. Gdy metadanych brakuje lub są niepełne, checkpointy wyglądające na
starsze checkpointy LibreYOLO są wczytywane ścieżką zgodności z ostrzeżeniem
i instrukcjami konwersji, natomiast obce checkpointy ze źródeł nadrzędnych są
kierowane do automatycznej konwersji. Zobacz
[checkpointy ze źródeł nadrzędnych](/docs/reference/upstream-checkpoints).

## Funkcje pomocnicze

Funkcje pomocnicze schematu znajdują się w `libreyolo.utils.serialization`:

```python
wrap_libreyolo_checkpoint(
    state_dict,
    *,
    model_family,
    size,
    task,
    nc,
    names=None,
    imgsz=None,
    libreyolo_version=None,
    schema_version="1.0",
    **extra_metadata,
) -> dict

validate_checkpoint_metadata(checkpoint, *, strict=False) -> list[str]

unwrap_libreyolo_checkpoint(loaded, *, strict=False) -> tuple[dict, dict]
```

`validate_checkpoint_metadata` nie modyfikuje danych i zwraca listę błędów.
Przy `strict=True` zamiast tego zgłasza `CheckpointMetadataError`.
`model.save(path)` jest obsługiwanym sposobem zapisu zgodnego checkpointu.

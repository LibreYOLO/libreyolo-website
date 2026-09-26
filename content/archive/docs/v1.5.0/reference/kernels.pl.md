---
title: Kernele
seo_title: Rejestr kerneli LibreYOLO i kernele Hub
description: >-
  Sposób wybierania przyspieszonych implementacji przez LibreYOLO: rejestr
  kerneli w libreyolo/kernels, opcjonalny kernel MS-deform-attn z Hugging Face
  Hub i przełącznik scalonego attention.
lead: >-
  Każda przyspieszona operacja w LibreYOLO ma przenośną implementację domyślną,
  a czasami również szybszy wariant zarejestrowany ponad nią. Wybór odbywa się w
  czasie działania za pomocą predykatu, brak opcjonalnej zależności powoduje
  użycie implementacji awaryjnej zamiast błędu, a wyeksportowany graf zawsze
  wybiera ścieżkę przenośną.
keywords:
  - kernele LibreYOLO
  - LIBREYOLO_KERNELS
  - LIBREYOLO_HUB_KERNELS
  - hub-kernels extra
  - kernel ms_deform_attn
  - set_fused_attention
  - kernele Triton LibreYOLO
last_verified: 1.5.0
verification: >-
  API rejestru odczytano z libreyolo/kernels/__init__.py w wersji 1.5.0, API
  attention z libreyolo/kernels/attention/__init__.py i sdpa.py, a dostawcę Hub
  z libreyolo/kernels/attention/ms_deform_attn.py wraz z przypiętą rewizją i
  predykatem kwalifikacji. Układ katalogów spisano z libreyolo/kernels/.
  Definicja dodatku pochodzi z pyproject.toml. Uwagi o zachowaniu i wyniki
  benchmarków pochodzą z docs/kernels.md. Historia bramkowania w v1.4.0 pochodzi
  z commita podłączającego slot RF-DETR oraz wpisu CHANGELOG dla wersji 1.5.0.
meta:
  - label: Pakiet
    value: libreyolo.kernels
    mono: true
  - label: Dodatek wymagający jawnego włączenia
    value: 'libreyolo[hub-kernels]'
    mono: true
  - label: Wymuszenie implementacji referencyjnej
    value: LIBREYOLO_KERNELS=off
    mono: true
snippets:
  usage:
    - label: Sprawdzenie wybranych implementacji
      language: python
      code: >
        import libreyolo.kernels as kernels


        # Mapowanie slotu operacji na nazwę wybranej implementacji lub
        "unavailable".

        print(kernels.active())
    - label: Wymuszenie ścieżki referencyjnej
      language: bash
      code: |
        # off i reference oznaczają to samo, a ponadto całkowicie pomijają
        # importowanie przyspieszonych dostawców.
        LIBREYOLO_KERNELS=off python train.py
    - label: Wyłączenie kerneli Hub bez odinstalowywania
      language: bash
      code: |
        LIBREYOLO_HUB_KERNELS=0 python predict.py
    - label: Przełączenie rodziny na scalone attention
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.kernels.attention import set_fused_attention

        model = LibreYOLO("LibreSwinIRs.pt")

        # Zwraca liczbę przełączonych modułów attention.
        print(set_fused_attention(model))
    - label: Rejestrowanie własnej implementacji
      language: python
      code: |
        import libreyolo.kernels as kernels

        kernels.register(
            "fake_quant_fp8",
            my_impl,
            name="mybackend",
            predicate=my_check,
        )
source_hash: 23d504e88b7959f8
---

## Rejestr

`libreyolo/kernels/` jest małym rejestrem wymiennych implementacji wybieranych
w czasie działania. Slot operacji to nazwa taka jak `fake_quant_fp8` lub
`ms_deform_attn`. Kod wywołujący prosi rejestr o slot i otrzymuje pierwszą
zarejestrowaną implementację, której predykat jest spełniony. Pierwszeństwo ma
najnowsza rejestracja, a gdy nic innego nie pasuje, wybierana jest implementacja
referencyjna.

Ta struktura istnieje po to, aby opcjonalna zależność nigdy nie była twardym
wymaganiem. Maszyna bez Triton, CUDA lub pakietu `kernels` wykonuje ten sam kod
i generuje te same wartości, tylko wolniej.

| Funkcja | Zastosowanie |
|---|---|
| `active()` | Mapowanie slotu operacji na nazwę wybranej implementacji lub `"unavailable"` |
| `resolve(op)` | Obiekt wywoływalny, który zostałby uruchomiony, albo `None` |
| `register(op, impl, *, name, predicate=None)` | Dodanie implementacji, najnowsza trafia na początek |
| `unregister(op, name)` | Usunięcie implementacji |
| `clear_cache()` | Usunięcie zapamiętanych rozstrzygnięć |

<code-tabs name="usage" />

Wyjątek z predykatu jest przechwytywany i powoduje ostrzeżenie, ale nigdy nie
jest przekazywany dalej. Wadliwa implementacja zewnętrzna przechodzi więc na
ścieżkę przenośną zamiast przerywać predykcję.

### Układ

Drzewo jest uporządkowane najpierw według zastosowania, a potem backendu,
dzięki czemu slot znajduje się według wykonywanych obliczeń, a nie biblioteki,
która akurat dziś go implementuje.

| Katalog | Zawartość |
|---|---|
| `kernels/quant/simulate/` | Kernele Triton symulowanej kwantyzacji, z prostym przebiegiem wstecz, na dowolnym urządzeniu. Używane zarówno przez QAT, jak i symulowaną kwantyzację po trenowaniu |
| `kernels/quant/execute/` | Ścieżki rzeczywistej precyzji tylko dla sfinalizowanych modeli, bez przebiegu wstecz: GEMM rdzeni tensorowych FP8, jego scalony prolog i epilog Triton oraz kernele rozpakowujące spakowane wagi |
| `kernels/attention/` | Operacje attention wspólne dla rodzin: slot `ms_deform_attn` i polityka scalonego SDPA |

Granicę między `simulate` i `execute` wyznacza finalizacja modelu, a nie jego
trenowanie lub wdrożenie. Implementacje referencyjne pozostają w
`libreyolo/quant/`, które definiuje znaczenie wartości. `kernels/` tylko
przyspiesza obliczenia. Pakowanie wag nie ma żadnych wariantów, ponieważ jest
częścią kontraktu checkpointu.

Sloty GEMM i attention nie mają implementacji referencyjnej. Kod wywołujący
musi sprawdzić, czy `resolve()` coś zwróciło, i zachować własną ścieżkę
przenośną, dlatego grafy ONNX, TensorRT i `torch.export` zawsze zawierają
przenośne obliczenia.

### Nadpisywanie wyboru

`LIBREYOLO_KERNELS=off` lub `=reference` wymusza implementacje referencyjne
i całkowicie pomija importowanie przyspieszonych dostawców. Każda inna wartość
ogranicza wybór do implementacji zarejestrowanych pod daną nazwą.
`LIBREYOLO_QUANT_KERNELS` jest obsługiwane jako starszy alias z czasów, gdy
rejestr znajdował się w `libreyolo/quant/`, i jest odczytywane tylko wtedy,
gdy `LIBREYOLO_KERNELS` nie jest ustawione. Obie zmienne wymieniono wraz
z pozostałymi na stronie [ustawień](/docs/reference/settings).

## Kernele Hub

Skompilowane kernele CUDA opublikowane w Hugging Face Hub są wczytywane podczas
działania przez opcjonalny pakiet `kernels`. Nic nie jest dołączane do
LibreYOLO. Pakiet pobiera artefakt i zapisuje go w pamięci podręcznej, a każdy
dostawca przypina sprawdzoną rewizję commita. Zmiana przypięcia wymaga więc
testu zgodności na GPU przed włączeniem.

Instalacja dodatku jest jawnym włączeniem:

```bash
pip install "libreyolo[hub-kernels]"
```

Bez pakietu nic się nie zmienia i nie jest wykonywane żądanie sieciowe.
`LIBREYOLO_HUB_KERNELS=0` wyłącza pobieranie bez odinstalowywania czegokolwiek.
Kernel, którego nie uda się wczytać lub uruchomić, wyłącza się do końca procesu
i po jednym ostrzeżeniu przechodzi na implementację awaryjną.

Obecnie Hub obsługuje jeden slot: `ms_deform_attn`, czyli skompilowany przebieg
w przód i wstecz wieloskalowego deformowalnego attention z Deformable DETR, na
licencji Apache 2.0. Jest podłączony do całej linii deformowalnej: RF-DETR,
Deformable DETR, DINO-DETR, LW-DETR, Grounding DINO, RT-DETR, RT-DETRv2, D-FINE,
RT-DETRv4, DEIM, DEIMv2, EC i OV-DEIM. Ponieważ przebieg wstecz jest również
skompilowany, zyskuje na tym zarówno trenowanie, jak i predykcja.

Kryteria kwalifikacji są celowo wąskie. Wejścia muszą korzystać z CUDA i float32,
a wykonanie musi odbywać się w trybie eager. Dostawca odmawia działania przy
`torch.jit.is_tracing()`, `torch.compiler.is_compiling()`,
`torch.compiler.is_exporting()` i `torch.onnx.is_in_onnx_export()`. Dwa układy
wejścia również przechodzą na ścieżkę przenośną: liczba punktów na poziom
różniąca się między poziomami oraz próbkowanie dyskretnych indeksów całkowitych.
Wariant EC pose nie jest podłączony.

### Ten kernel stał się właśnie dostępny

Przed zainstalowaniem dodatku w istniejącym projekcie należy przeczytać tę
sekcję.

W v1.4.0 slot był sprawdzany wewnątrz funkcji pomocniczej, za warunkiem
wymagającym braku par kształtów przestrzennych. RF-DETR zawsze przekazuje te
pary przez dekoder, więc warunek nigdy nie był spełniony, a kernel nie wykonywał
się w żadnym przebiegu w przód w trybie eager. W v1.5.0 przeniesiono punkt
sprawdzania i kernel rzeczywiście działa.

W praktyce aktualizacja do v1.5.0 połączona z instalacją
`libreyolo[hub-kernels]` na CUDA oznacza, że RF-DETR i powiązane rodziny po raz
pierwszy wykonują przebieg w przód ze skompilowanego pliku binarnego. Z tego
powodu predykcje i metryki mogą zmienić się w granicach tolerancji
zmiennoprzecinkowej. Standardowa instalacja bez dodatku pozostaje bez zmian.
Podczas porównywania metryk między wersjami należy zachować ten sam stan dodatku
lub ustawić `LIBREYOLO_HUB_KERNELS=0` po obu stronach.

## Scalone attention

Scalone attention iloczynu skalarnego ze skalowaniem nie wymaga opcjonalnej
zależności, tylko standardowego PyTorch, dlatego podlega polityce, a nie
dostępności. Obowiązują dwie reguły.

Po pierwsze, przechwytywanie grafu nigdy go nie używa. Każde podmienione miejsce
wywołania zachowuje równanie operacji pierwotnych za kontrolą eksportu. Obejmuje
to eksport ONNX, którego domyślny opset nie ma symbolu SDPA, oraz
`torch.jit.trace`, używane przez TorchScript, CoreML i NCNN. Przechwytywanie
Dynamo celowo pozostaje poza bramką, ponieważ `torch.compile` obniża SDPA lepiej
niż ręczne obliczenia, a Core AI i ExecuTorch samodzielnie rozkładają SDPA na
podstawowe operacje ATen.

Po drugie, próg włączenia jako wartości domyślnej wymaga zgodności bajtowej.
Rodziny, które go spełniają, domyślnie używają SDPA: SegFormer, Depth Anything
i MoGe-2, BERT, Grounding DINO, SwinIR i PP-OCR. Rodziny, które go nie
spełniają, zachowują ręczne obliczenia i udostępniają zamiast tego flagę
`fused_attn`, przełączaną przez `set_fused_attention(model)`: Swin, backbone
Swin modelu DINO-DETR, BiRefNet i FeyNobg, OWLv2, LW-DETR, SigLIP 2, ZipDepth
i MobileSAM. ViT i DeiT mają tę samą flagę, ale zgodnie ze źródłem jest ona
domyślnie włączona, więc to samo wywołanie z `enabled=False` ją wyłącza.

Tam, gdzie ma zastosowanie, warto to zrobić. Na RTX 5070 Ti przy automatycznym
rzutowaniu fp16 czas attention okna Swin spada z 1.278 ms do 0.721 ms, co daje
przyspieszenie 1.77x, a czas attention części wizyjnej OWLv2 spada z 6.483 ms
do 1.735 ms, czyli 3.74x.

## Sprzęt

| Platforma | Zachowanie |
|---|---|
| CPU i MPS | Każdy predykat CUDA i Triton jest niespełniony, więc wszystko działa na implementacji referencyjnej |
| NVIDIA CUDA | Włączają się kernele Triton oraz kwalifikujące się kernele Hub i GEMM |
| AMD ROCm | Triton może się włączyć, ponieważ pakiety ROCm zawierają backend AMD dla Triton, ale zgodność jest testowana w CI tylko na NVIDIA |

## Dodawanie implementacji

Wywołaj `register()` z nazwą i predykatem. Zewnętrzne skompilowane kernele mogą
być dostarczane jako osobny pakiet `libreyolo_kernels`, który rejestruje się
podczas importu. Dzięki temu prywatny backend pozostaje całkowicie poza drzewem
LibreYOLO.

Warunkiem włączenia czegokolwiek do repozytorium jest zgodność: dokładna
zgodność przebiegu w przód z implementacją referencyjną oraz gradienty
odbiegające o nie więcej niż 1e-6 od estymatora prostego, dla zestawu kształtów
zawartego w testach.

Wybór kernela współdziała z [grafami CUDA](/docs/reference/cuda-graphs): macierz
zgodności inferencji działała bez zainstalowanego pakietu `kernels`, dlatego nie
obejmuje bezpieczeństwa przechwytywania przy aktywnym skompilowanym kernelu.

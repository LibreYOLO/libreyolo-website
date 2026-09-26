---
title: Dostrajanie LoRA
seo_title: Dostrajanie LoRA w LibreYOLO
description: >-
  Dostrajaj detektor transformerowy przy małej ilości VRAM za pomocą lora=True.
  Poznaj dziewięć obsługiwanych rodzin, recepturę adapterów każdej rodziny i
  zachowanie checkpointów.
lead: >-
  LoRA zamraża duże, wstępnie wytrenowane części modelu i trenuje obok nich małe
  adaptery niskiego rzędu oraz warstwy, które muszą pozostać gęste. W LibreYOLO
  cały publiczny interfejs sprowadza się do jednej wartości logicznej.
keywords:
  - dostrajanie lora
  - parameter efficient fine tuning
  - peft
  - dora
  - trenowanie przy małej pamięci vram
  - rf-detr lora
  - d-fine lora
  - scalanie adapterów
last_verified: 1.5.0
snippets:
  install:
    - label: pip
      language: bash
      code: |
        pip install "libreyolo[lora]"
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 lora=true
  merge:
    - label: Eksport ze scalaniem adapterów
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("runs/train/exp/weights/best.pt")
        model.export(format="onnx")
    - label: Scalanie w miejscu
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training.lora import merge_lora_adapters

        model = LibreYOLO("runs/train/exp/weights/best.pt")
        merged = merge_lora_adapters(model.model)

        print(f"{merged} adapter layers folded into dense weights")
source_hash: 603fdddf5ec0c316
---

## Instalacja

LoRA korzysta z opcjonalnej zależności `peft`.

<code-tabs name="install" />

Bez niej `lora=True` zgłasza `ImportError` wskazujący to polecenie, zamiast
przypadkowo uruchomić pełne dostrajanie.

## Używanie

<code-tabs name="train" />

`lora=True` jest całym interfejsem. Rząd, alfa, dropout i moduły docelowe są
ustalone dla każdej rodziny zgodnie z jej materiałem nadrzędnym i nie są
parametrami dostępnymi dla użytkownika.

Rodzina bez obsługi LoRA zgłasza błąd podczas konfiguracji zamiast ignorować
flagę:

```text
LoRA fine-tuning (lora=True) is not supported for yolo9. LoRA targets
transformer components with nn.Linear layers (e.g. RF-DETR, D-FINE, DEIM).
```

CLI odrzuca ją wcześniej, przed zbudowaniem modelu, używając własnej listy tych
samych dziewięciu dozwolonych rodzin.

## Obsługiwane rodziny

RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETR v1, v2 i v4, EC oraz ConvNeXt. Bramą jest
atrybut `supports_lora` w klasie trenera każdej rodziny, a CLI zawiera zgodną
listę dozwolonych rodzin.

Zakres zadań jest węższy niż zakres rodzin. D-FINE i EC obsługują tylko
detekcję, a ich ścieżki segmentacji i pozy zgłaszają błąd. Ścieżka semantyczna
RF-DETR zgłasza błąd. ConvNeXt służy do klasyfikacji.

Wszystkie pozostałe przypadki zgłaszają błąd. Nie ma trybu częściowego ani
cichego.

## Działanie poszczególnych receptur

Receptury różnią się z powodu odmiennych architektur. Receptura działająca na
backbone ViT nie ma do czego dołączyć się w modelu splotowym.

RF-DETR używa DoRA, czyli LoRA z dekompozycją wag, o rzędzie 16 i wartości alfa
16 na projekcjach uwagi `query`, `key` oraz `value` w backbone DINOv2, zgodnie z
materiałem referencyjnym RF-DETR. Backbone ViT zostaje zamrożony, natomiast
projektor, dekoder i głowica detekcji nadal są trenowane zwykłym sposobem.

D-FINE, DEIM oraz RT-DETR v1, v2 i v4 łączą splotowy backbone z hybrydowym
enkoderem transformerowym i dekoderem deformowalnym, dlatego podział jest inny.
Splotowy backbone zostaje całkowicie zamrożony, co pomija także jego przebieg
wsteczny. Bloki transformera zamrażają wagi podstawowe i trenują zwykłe adaptery
LoRA o tym samym rzędzie 16 i wartości alfa 16 na warstwach liniowych:
feed-forward `linear1` i `linear2`, bramie oraz projekcjach uwagi deformowalnej.
Wszystkie pozostałe elementy, czyli splotowa fuzja enkodera, projekcje wejściowe,
głowice predykcji i embeddingi zapytań, nadal są trenowane gęsto.

Dwa szczegóły tej receptury są celowe. Mechanizm self-attention dekodera
pozostaje zamrożony bez adapterów, ponieważ `nn.MultiheadAttention` z PyTorch
odczytuje `out_proj.weight` bezpośrednio i po cichu ominąłby wstrzyknięty adapter.
Stosowana jest zwykła LoRA, a nie DoRA, ponieważ kilka warstw liniowych dekodera
jest z założenia inicjalizowanych zerami, natomiast normalizacja wielkości DoRA
dzieli przez normę wag.

DEIMv2 używa tej samej receptury z warstwami feed-forward SwiGLU `w12` i `w3`
jako celami. Rozmiary S, M, L oraz X zawierają także backbone ViT DINOv3, w
którym podstawa ViT zostaje zamrożona, a jej scalone warstwy uwagi `qkv`
otrzymują adaptery. Jednocześnie piramida splotowa Spatial Tuning Adapter jest
nadal trenowana jako odpowiednik projektora. Adaptery `qkv` są dodawane nawet
wtedy, gdy konfiguracja zawierała zamrożony ViT, ponieważ celem jest właśnie
dostosowanie zamrożonego backbone. Rozmiary mniejsze niż S używają splotowego
backbone i zwykłej receptury.

EC jest detektorem DETR, którego backbone stanowi ViT otoczony możliwą do
trenowania piramidą projektora splotowego. Podstawa ViT zostaje zamrożona, a jej
warstwy `qkv` otrzymują adaptery. Bloki transformera używają wspólnej receptury,
natomiast projektor i głowice pozostają gęste.

Bloki ConvNeXt zawierają liniowe moduły MLP z kanałami na końcu, `fc1` i `fc2`,
które otrzymują zwykłe adaptery. Sploty głębokie, normalizacje i parametry skali
warstw zostają zamrożone. Głowica klasyfikacji pozostaje gęsta, aby nadal działały
niestandardowe liczby klas.

Głowice detekcji i klasyfikacji zawsze pozostają możliwe do trenowania we
wszystkich recepturach, ponieważ niestandardowa liczba klas wymaga świeżo
wytrenowanej głowicy.

## Checkpointy i eksport

`best.pt` i `last.pt` zachowują tensory adapterów, dlatego przebieg LoRA można
wznowić lub sprawdzić tak jak każdy inny. Wczytanie jednego z tych checkpointów
wymaga zainstalowanego dodatku `lora`, ponieważ moduł wczytujący powtarza
wstrzyknięcie adapterów, aby klucze były zgodne.

`export()` scala adaptery z gęstymi wagami, dlatego wyeksportowany artefakt nie
zależy od `peft`. To samo scalanie jest dostępne bezpośrednio dla modelu w
pamięci.

<code-tabs name="merge" />

Po scaleniu drzewo modułów jest w pełni gęste, a drugie scalanie nie wykonuje
żadnej operacji.

## Oszczędności i ograniczenia

LoRA zmniejsza zużycie pamięci optymalizatora i gradientów. W rodzinach, które
całkowicie zamrażają backbone, pomija również jego przebieg wsteczny.

Pamięć aktywacji pozostaje bez zmian. Aktywacje z przebiegu do przodu nadal
trzeba zachować dla wszystkich elementów możliwych do trenowania, a zwykle to
one wyznaczają wartość szczytową. Przy najbardziej ograniczonym budżecie VRAM
należy również zmniejszyć `batch` lub `imgsz`.

## Powiązane strony

- [Zamrażanie warstw](/docs/train/layer-freezing) opisuje drugi sposób trenowania
  podzbioru wag, który działa w każdej rodzinie i nie wymaga dodatkowej
  zależności. `freeze` i `lora=True` można łączyć. Parametry adapterów pozostają
  możliwe do trenowania nawet po zamrożeniu nadrzędnej grupy backbone.
- [Hiperparametry](/docs/train/hyperparameters) opisują `batch`, `imgsz` i
  pozostałe argumenty `train()`.


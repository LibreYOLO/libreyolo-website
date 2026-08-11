---
title: Zamrażanie warstw
seo_title: Zamrażanie warstw podczas trenowania w LibreYOLO
description: >-
  Zamrażaj część modelu podczas uczenia transferowego: podaj całkowitą liczbę
  grup zamrażania rodziny, jawną listę indeksów albo selektory nazw modułów i
  parametrów.
lead: >-
  Zamrażanie utrzymuje wybrane wagi bez zmian podczas trenowania pozostałej
  części modelu. Selektory odnoszą się do uporządkowanych grup zamrażania danej
  rodziny lub nazw jej modułów, a nie do surowych numerów warstw z grafu YAML.
keywords:
  - zamrażanie warstw
  - uczenie transferowe
  - zamrażanie backbone
  - zamrożony batchnorm
  - grupy zamrażania
  - dostrajanie tylko głowicy
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Pierwsze 10 grup obejmuje cały backbone YOLOv9.
        model.train(data="my-dataset.yaml", epochs=50, freeze=10)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=50 freeze=10
    - label: Według nazwy
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, freeze="backbone")
    - label: Kilka selektorów
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", freeze=["backbone", "neck"])
  groups:
    - label: Wyświetlanie grup zamrażania rodziny w kolejności
      language: python
      code: |
        from libreyolo import LibreYOLO9
        from libreyolo.models.yolo9.trainer import YOLO9Trainer

        model = LibreYOLO9("LibreYOLO9s.pt", size="s")
        trainer = YOLO9Trainer(model=model.model, wrapper_model=model, size="s")

        for index, (name, _module) in enumerate(trainer.get_freeze_groups()):
            print(index, name)
source_hash: 9f1e7551af6b16fe
---

## Zamrażanie elementów

`freeze` jest opcjonalne i domyślnie nie zamraża niczego.

<code-tabs name="train" />

Zamrażanie odbywa się po zbudowaniu modelu i ewentualnej przebudowie głowicy dla
nowej liczby klas, ale przed utworzeniem optymalizatora. Optymalizator zawsze
otrzymuje więc wyłącznie parametry możliwe do trenowania.

## Możliwe postacie selektora

| Wartość | Znaczenie |
|---|---|
| `None`, `False`, `""`, `"none"` | Trenowanie każdego parametru |
| `10` lub `"10"` | Zamrożenie pierwszych dziesięciu grup zamrażania rodziny |
| `[0, 3, 7]` | Zamrożenie wskazanych grup numerowanych od zera |
| `"backbone"` | Zamrożenie pasującej grupy, modułu lub prefiksu parametru |
| `["backbone", "neck"]` | Zamrożenie każdego wymienionego selektora |
| `["backbone", 3]` | Listy mieszane są obsługiwane |

Ciąg znaków jest analizowany przed interpretacją, dlatego CLI i konfiguracja
YAML przyjmują te same postacie co Python. `freeze="[0, 3, 'head']"` jest
analizowane jako dosłowna lista, `freeze="backbone,neck"` jest dzielone przy
przecinku, a sam ciąg cyfr dziesiętnych staje się liczbą grup.

`freeze=True` jest odrzucane jako niejednoznaczne.

Selektory nazw dopasowują nazwę grupy zamrażania, nazwę modułu lub prefiks nazwy
parametru. Działają znaki glob `*`, `?` i `[`. Początkowe `model.` jest
traktowane elastycznie, więc `backbone` i `model.backbone` trafiają w pisownię
używaną wewnętrznie przez rodzinę.

## Grupy definiuje rodzina

Liczba całkowita odnosi się do własnej uporządkowanej listy grup zamrażania danej
rodziny, a nie do pozycji we wspólnym grafie. Nie wszystkie rodziny LibreYOLO są
jednym modelem sekwencyjnym indeksowanym przez YAML, dlatego surowy numer warstwy
miałby w każdej z nich inne znaczenie.

YOLOv9 porządkuje grupy od strony wejścia: dziesięć etapów backbone, następnie
sześć etapów neck i na końcu głowica. Dlatego `freeze=10` oznacza dokładnie
backbone. Stabilnymi selektorami nazw są dodatkowo `backbone`, `neck` i `head`.

Grupy RF-DETR to `backbone.encoder`, `backbone.projector`, `decoder`, `queries`,
`transformer.encoder_output` oraz `head`. Nazwy są tutaj lepszym wyborem,
ponieważ elementy transformera nie odpowiadają liczbie warstw. `backbone`
dopasowuje obie grupy backbone według prefiksu.

Rodziny, które nie definiują grup semantycznych, używają zachowawczej wartości
rezerwowej: każde bezpośrednie dziecko modelu mające co najmniej jeden parametr,
w kolejności deklaracji. Zwykle jest to krótka lista, dlatego duża liczba
całkowita nie znajdzie wystarczającej liczby grup:

```text
freeze index 10 is out of range for 3 available freeze groups.
```

Aby zobaczyć rzeczywistą listę zamiast zgadywać:

<code-tabs name="groups" />

## Błędy są wyraźnie zgłaszane

Każdy błędny wariant zgłasza wyjątek, zamiast trenować coś, czego nie zażądano.

Selektor, który niczego nie dopasowuje, zgłasza błąd i wymienia nietrafione
selektory:

```text
freeze selector(s) matched no parameters: 'backbon'
```

Zamrożenie, które nie pozostawiłoby żadnych parametrów możliwych do trenowania,
zgłasza błąd zarówno podczas zamrażania, jak i ponownie podczas budowania
optymalizatora:

```text
freeze would leave no trainable parameters. Use a smaller freeze value or
target a narrower module.
```

Tak właśnie działa `freeze="all"`, ponieważ `all` dopasowuje każdy parametr.

Po udanym zamrożeniu jeden wiersz rejestruje wykonane działanie:

```text
Layer freezing: selectors=[10], tensors=124, params=2103776, trainable=1863456/3967232
```

## Zamrożony BatchNorm przestaje się aktualizować

Zamrożony parametr nadal znajduje się w module, którego statystyki bieżące
mogłyby się zmieniać. Każdy moduł w stylu BatchNorm, którego parametry trafiły do
zamrożonego zestawu, jest przełączany w tryb ewaluacji. Trener stosuje to
ponownie po każdym wywołaniu `model.train()` w epoce, dzięki czemu statystyki
pozostają stałe przez cały przebieg.

Jest to zachowanie domyślne, dzięki któremu zamrożenie backbone faktycznie go
zamraża.

## Łączenie z LoRA

`freeze` i `lora=True` działają razem. W RF-DETR, DEIM i ConvNeXt parametry
adapterów pozostają możliwe do trenowania nawet po zamrożeniu grupy nadrzędnej.
Jest to pożądana kombinacja: zamrożony backbone z uczącymi się na nim adapterami.
Zobacz [dostrajanie LoRA](/docs/train/lora).

## Zakres

Jest to statyczne zamrażanie ustalane podczas uruchamiania. Planowane odmrażanie
i stopniowe zamrażanie nie należą do interfejsu.

## Powiązane strony

- [Hiperparametry](/docs/train/hyperparameters) opisują pozostałe argumenty
  `train()`.
- [Destylacja](/docs/train/distillation) opisuje drugi sposób przenoszenia wiedzy
  dużego modelu do przebiegu trenowania.


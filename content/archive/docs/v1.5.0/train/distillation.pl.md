---
title: Destylacja wiedzy
seo_title: Destylacja wiedzy w LibreYOLO
description: >-
  Trenuj mały detektor z większym nauczycielem lub zamrożonym backbone DINOv2:
  poznaj funkcje straty MGD, CWD i feature-MSE, punkty przechwytywania oraz
  obsługiwane rodziny.
lead: >-
  Destylacja dodaje drugi składnik funkcji straty, który zbliża pośrednie mapy
  cech ucznia do map zamrożonego nauczyciela. LibreYOLO przechwytuje cechy za
  pomocą forward hooks, dlatego głowica i funkcja straty nauczyciela nigdy nie
  uczestniczą w procesie.
keywords:
  - destylacja wiedzy
  - masked generative distillation
  - channel-wise distillation
  - destylacja cech
  - nauczyciel dinov2
  - trenowanie teacher student
  - funkcja straty mgd
  - funkcja straty cwd
last_verified: 1.5.0
snippets:
  detector:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Większy checkpoint tej samej rodziny nadzoruje mniejszy model.
        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            distill_model="LibreYOLO9c.pt",
            distill_loss_type="mgd",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 distill_model=LibreYOLO9c.pt distill_loss_type=mgd
  foundation:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Zamrożony samonadzorowany ViT nadzoruje jeden etap backbone.
        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            distill_model="dinov2",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 distill_model=dinov2
  tuned:
    - label: Dostrajanie funkcji straty
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            distill_model="LibreYOLO9c.pt",
            distill_loss_type="cwd",
            dis=1.0,           # globalna waga destylacji
            distill_tau=1.0,   # temperatura softmax CWD
        )
source_hash: 7210031328f6826f
---

## Destylacja z większego checkpointu

Ustawienie `distill_model` włącza destylację. Wartością jest checkpoint
nauczyciela wczytywany przez tę samą fabrykę co każdy inny model.

<code-tabs name="detector" />

Nauczyciel wykonuje przebieg w ramach `no_grad` oraz autocast, gdy AMP jest
włączone, dzięki czemu zamrożony model nie ponosi kosztu obliczeń z pełną
precyzją w każdym kroku. Forward hooks przechwytują jego mapy cech w nazwanych
punktach, funkcja straty porównuje je z mapami ucznia, a wynik jest dodawany do
funkcji straty trenowania i raportowany jako składnik o nazwie `distill`.

## Destylacja z zamrożonego modelu bazowego backbone

Samonadzorowany ViT może zamiast tego nadzorować pojedynczy etap backbone ucznia.
Cechy nauczyciela pochodzą z jego własnego ekstraktora cech, a nie z hooks, zaś
funkcja straty obsługuje różnicę między siatką patchy a krokiem splotowym.

<code-tabs name="foundation" />

`distill_model` rozpoznaje `dinov2`, oznaczające DINOv2-base, a także
`dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, `dinov2-small`,
`dinov2-base`, `dinov2-large` i każdy surowy identyfikator w hubie zaczynający
się od `facebook/dinov2`. Każda inna wartość jest traktowana jako ścieżka
checkpointu nauczyciela.

Ta ścieżka używa `feat_mse` niezależnie od `distill_loss_type` i wymaga
zainstalowanego `transformers`. Jeśli nauczyciel zostanie wczytany z brakującymi
kluczami wag, proces zostanie przerwany zamiast prowadzić destylację względem
częściowo losowego backbone.

## Obsługiwane rodziny

Obsługa destylacji jest metodą modelu ucznia i istnieją dwie takie metody.

`get_distill_config()` udostępnia wieloskalowe punkty przechwytywania nadzorowane
przez nauczyciela detektora. Implementują ją YOLOv9, YOLOX i RF-DETR.

`get_backbone_distill_config()` udostępnia pojedynczy etap backbone nadzorowany
przez nauczyciela bazowego. Implementuje ją YOLOv9 i jest jedyną rodziną, która
to robi.

Każda inna rodzina zgłasza błąd zamiast trenować bez tej funkcji straty:

```text
LibreDFINE does not implement get_distill_config(). Distillation is not yet
supported for the 'dfine' family.
```

```text
Foundation-model distillation into the 'yolox' family is not supported yet
(no get_backbone_distill_config()).
```

## Punkty przechwytywania

Punkty przechwytywania są stałe dla każdej rodziny i roli. Nauczyciel i uczeń
nie muszą więc mieć tej samej architektury, ale muszą mieć zgodne kroki cech.

| Rodzina | Rola | Punkty przechwytywania | Kroki |
|---|---|---|---|
| YOLOv9 | nauczyciel lub uczeń | `neck.elan_up2`, `neck.elan_down1`, `neck.elan_down2` | 8, 16, 32 |
| YOLOv9 | uczeń modelu bazowego | `backbone.elan3` | 16 |
| YOLOX | nauczyciel lub uczeń | `backbone.C3_p3`, `backbone.C3_n3`, `backbone.C3_n4` | 8, 16, 32 |
| RF-DETR | nauczyciel lub uczeń | `model.backbone.0.projector.stages.0` | sprawdzane podczas konfiguracji |

Niezgodne kroki powodują błąd przed rozpoczęciem trenowania:

```text
Teacher and student must have matching strides. Teacher: [8, 16, 32],
Student: [16]
```

To sprawdzenie jest pomijane dla nauczycieli bazowych, których istotą jest
różnica między siatkami.

## Trzy funkcje straty

`distill_loss_type` wybiera funkcję straty cech dla nauczyciela detektora.
Nauczyciel bazowy zawsze używa `feat_mse`.

`mgd`, czyli masked generative distillation, maskuje część pozycji przestrzennych
ucznia i trenuje mały generator z dwoma splotami, aby z pozostałych pozycji
odtworzył pełną mapę cech nauczyciela. `distill_mask_ratio` określa maskowaną
część, domyślnie 0.65.

`cwd`, czyli channel-wise distillation, przekształca aktywacje przestrzenne
każdego kanału w rozkład prawdopodobieństwa i minimalizuje dywergencję KL kanał
po kanale. `distill_tau` jest temperaturą softmax, domyślnie 1.0.

`feat_mse` wyrównuje kanały ucznia z kanałami nauczyciela za pomocą splotu 1x1,
zmienia biliniowo rozmiar siatki nauczyciela do siatki ucznia i oblicza średni
błąd kwadratowy. `distill_normalize=True` najpierw normalizuje L2 obie mapy cech
wzdłuż wymiaru kanałów, dzięki czemu dopasowanie zależy wyłącznie od kąta i jest
niezmienne względem skali. Wartość domyślna to `False`.

`dis` jest globalną wagą stosowaną dodatkowo. Jeśli nie zostanie ustawione,
każda funkcja straty używa własnej opublikowanej wartości domyślnej: 2e-5 dla
MGD, 1.0 dla CWD i 1.0 dla feature MSE. Wartości te różnią się o pięć rzędów
wielkości, dlatego waga dostrojona dla jednego rodzaju funkcji straty jest
bezużyteczna dla innego.

<code-tabs name="tuned" />

`distill_mask_ratio`, `distill_tau` i `distill_normalize` nie mają flag CLI. Są
argumentami Pythona lub kluczami YAML w `cfg=`. W przypadku RF-DETR cała
destylacja jest również dostępna tylko w Pythonie, ponieważ mapowanie argumentów
CLI tej rodziny nie zawiera kluczy destylacji.

## Adaptery, checkpointy i wiele GPU

Każda funkcja straty buduje małe moduły możliwe do trenowania, które znajdują
się poza uczniem: adaptery kanałów 1x1 oraz generator MGD. Otrzymują własną grupę
parametrów optymalizatora z efektywnym współczynnikiem uczenia przebiegu.

Moduły są zapisywane w checkpoincie pod kluczem `distiller` i przywracane po
wznowieniu, dlatego wznowiony przebieg nie zaczyna z zimnymi projektorami.

W DDP adaptery znajdują się poza opakowanym uczniem, co oznacza, że reduktor DDP
nigdy nie widzi ich gradientów. Trener jawnie wykonuje dla nich all-reduce w
każdym kroku, dzięki czemu każdy proces trenuje te same adaptery.

Przechwytywanie grafu CUDA nie jest dostępne w przebiegu destylacji. Przekazanie
`cuda_graph=True` zapisuje jeden wiersz w logu i prowadzi trenowanie w trybie
eager. Zobacz [wydajność trenowania](/docs/train/performance).

## Powiązane strony

- [Zamrażanie warstw](/docs/train/layer-freezing) oraz
  [dostrajanie LoRA](/docs/train/lora), z których żadna funkcja nie blokuje
  łączenia z destylacją.
- [Hiperparametry](/docs/train/hyperparameters) opisują pozostałe argumenty
  `train()`.


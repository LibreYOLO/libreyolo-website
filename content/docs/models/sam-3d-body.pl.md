---
title: SAM 3D Body
families:
  - sam3dbody
seo_title: 'SAM 3D Body: odtwarzanie siatki całego ciała w LibreYOLO'
description: >-
  Używaj SAM 3D Body w LibreYOLO do odtwarzania siatki 3D całego ciała
  człowieka. Instaluj i przewiduj. Checkpointy są kontrolowane licencją SAM
  License firmy Meta i wymagają CUDA.
lead: >-
  SAM 3D Body to sterowany promptami model Meta do odtwarzania siatki 3D całego
  ciała, w tym dłoni i stóp, z jednego obrazu i ramek osób. LibreYOLO opakowuje
  pakiet źródłowy zamiast go przenosić.
keywords:
  - SAM 3D Body
  - odtwarzanie siatki człowieka
  - siatka ciała
  - MHR
  - Momentum Human Rig
  - estymacja pozy 3D
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import SAMPLE_IMAGE

        from libreyolo.models.sam3dbody import LibreSAM3DBody


        # Ta rodzina nie jest zarejestrowana w fabryce LibreYOLO(), dlatego

        # tworzy się ją bezpośrednio. model_path=None uruchamia kontrolowane

        # pobieranie z Hugging Face. Ciąg jest natomiast traktowany jako

        # istniejąca ścieżka lokalnego checkpointu i nigdy nie jest pobierany
        automatycznie.

        # Inferencja wymaga urządzenia CUDA. Nie ma ścieżki CPU.

        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])


        meshes = result.meshes

        print(meshes.vertices.shape)    # (N, V, 3), układ kamery, metry

        print(meshes.joints3d.shape)    # (N, J, 3)
    - label: Z detektorem osób
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # Nie ma skrótu w postaci nazwanego ciągu. Przekaż utworzony detektor
        # LibreYOLO, zwykły obiekt wywoływalny albo instancję PersonDetector.
        detector = LibreYOLO("LibreRFDETRn.pt")
        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_detector=detector)
source_hash: 8edc8d7872f3f875
---

## Instalacja

```bash
pip install libreyolo
```

Instaluje to tylko adapter LibreYOLO. Sam SAM 3D Body nie jest dołączony,
ponieważ jego licencja nie zezwala na tworzenie na jej podstawie własnego kodu
LibreYOLO. Należy sklonować repozytorium źródłowe, samodzielnie zainstalować
jego zależności, a następnie wskazać klon LibreYOLO.

```bash
git clone https://github.com/facebookresearch/sam-3d-body
pip install roma einops yacs omegaconf braceexpand pytorch-lightning timm
```

```python
from libreyolo.models.sam3dbody import LibreSAM3DBody

model = LibreSAM3DBody(
    None,
    size="d3",
    sam_3d_body_path="/path/to/sam-3d-body",
    device="cuda",
)
```

Można też ustawić zmienną środowiskową `SAM_3D_BODY_PATH` zamiast przekazywać
`sam_3d_body_path` przy każdym wywołaniu. Użytkownik, który nigdy nie tworzy tej
rodziny, nigdy nie uruchamia importu ani nie styka się z SAM License. Ta rodzina
nie jest podłączona do fabryki `LibreYOLO()` ani polecenia CLI
`libreyolo predict`. `LibreSAM3DBody` jest jedynym punktem wejścia.

## Predykcja

<code-tabs name="predict" />

Dostęp do pobierania checkpointu jest kontrolowany. Wymaga zaakceptowania
licencji Meta na stronie modelu Hugging Face i uwierzytelnienia przez
`hf auth login` przed pierwszym pobraniem. Sama inferencja zawsze wymaga
urządzenia CUDA. Estymator źródłowy przenosi batch na GPU bez sprawdzania,
dlatego komputer bez GPU zgłasza błąd zamiast przełączać się na CPU.
`result.meshes` jest elementem `Meshes`, wyrównanym wierszami z `result.boxes`
(jeden wiersz na wykrytą osobę). `vertices` i `joints3d` mają wartości metryczne
i zawierają już oszacowane przesunięcie kamery, `joints2d` podaje piksele
oryginalnego obrazu, a obroty używają konwencji MHR, czyli kątów Eulera zamiast
reprezentacji oś-kąt. Więcej informacji o źródłach, streamingu i obsłudze
wyników zawiera strona [predykcji](/docs/predict).

## Warianty

Dwa backbone współdzielą ten sam model ciała MHR. `d3` używa enkodera DINOv3
ViT-H/16+, a `h` oryginalnego enkodera ViT-H.

## Eksport

<export-matrix />

Eksport siatki ciała nie jest zaimplementowany. LibreYOLO nie zdefiniowało
jeszcze kontraktu wyeksportowanego grafu dla zadania siatki, w tym sposobu
reprezentowania układu parametrów MHR poza PyTorch.

## Checkpointy

Wszystkie opublikowane pliki wag dla tej rodziny.

<checkpoint-table />

## Licencja

<provenance-box>

Model ciała sterowany przez checkpointy, MHR (Momentum Human Rig), jest osobnym
wydaniem Meta na licencji Apache-2.0. LibreYOLO podczas działania pobiera jego
artefakt TorchScript z własnego publicznego wydania MHR i zapisuje go lokalnie
w pamięci podręcznej. LibreYOLO nie tworzy kopii lustrzanej tego pliku. Podlega
on własnym warunkom Apache-2.0, a nie SAM License.

</provenance-box>

## Cytowanie

<citation-block />

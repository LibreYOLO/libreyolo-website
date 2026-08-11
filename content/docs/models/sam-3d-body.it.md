---
title: SAM 3D Body
families:
  - sam3dbody
seo_title: 'SAM 3D Body: recupero della mesh di corpo intero in LibreYOLO'
description: >-
  Usa SAM 3D Body in LibreYOLO per il recupero della mesh 3D di un corpo umano
  intero. Installa e fai predizioni; la SAM License di Meta vincola i checkpoint
  e serve CUDA.
lead: >-
  SAM 3D Body è il modello di Meta guidato da prompt per recuperare una mesh 3D
  di corpo intero, mani e piedi inclusi, a partire da una singola immagine e da
  box di persone. LibreYOLO fa da wrapper al pacchetto upstream invece di
  portarlo.
keywords:
  - SAM 3D Body
  - MHR
  - Momentum Human Rig
  - human mesh recovery
  - mesh 3d corpo umano
  - ricostruzione 3d persone
  - posa 3d python
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # Questa famiglia non è registrata nella factory LibreYOLO(),
        # quindi la si costruisce direttamente. model_path=None è ciò che
        # innesca il download vincolato da Hugging Face; una stringa è
        # invece trattata come il percorso di un checkpoint locale già
        # esistente e non viene mai scaricata automaticamente.
        # L'inferenza richiede un dispositivo CUDA; non c'è un percorso CPU.
        model = LibreSAM3DBody(None, size="d3", device="cuda")
        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])

        meshes = result.meshes
        print(meshes.vertices.shape)    # (N, V, 3), sistema della camera, metri
        print(meshes.joints3d.shape)    # (N, J, 3)
    - label: Con un detector di persone
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # Qui non c'è una scorciatoia con stringa nominata: passa un
        # detector LibreYOLO già costruito, un semplice callable oppure
        # un'istanza di PersonDetector.
        detector = LibreYOLO("LibreRFDETRn.pt")
        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_detector=detector)
source_hash: 8edc8d7872f3f875
---

## Installazione

```bash
pip install libreyolo
```

Questo ti dà solo l'adattatore di LibreYOLO. SAM 3D Body in sé non è incluso,
perché la sua licenza non è una da cui il codice di LibreYOLO possa derivare:
clona il repository upstream e installa le sue dipendenze per conto tuo, poi fai
puntare LibreYOLO al clone.

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

oppure imposta la variabile d'ambiente `SAM_3D_BODY_PATH` invece di passare
`sam_3d_body_path` a ogni chiamata. Chi non costruisce mai questa famiglia non
innesca mai l'import, e non incontra mai la SAM License. Questa famiglia non è
collegata alla factory `LibreYOLO()` né al comando CLI `libreyolo predict`;
`LibreSAM3DBody` è l'unico entry point.

## Predizione

<code-tabs name="predict" />

Il download del checkpoint è vincolato: richiede di accettare la licenza di Meta
sulla pagina del modello su Hugging Face e di autenticarsi con `hf auth login`
perché il primo download vada a buon fine. L'inferenza stessa ha bisogno di un
dispositivo CUDA senza eccezioni: l'estimatore upstream sposta il suo batch sulla
GPU senza controllare, quindi una macchina con la sola CPU solleva un errore
invece di ricadere su un fallback. `result.meshes` è un payload `Meshes`,
allineato riga per riga con `result.boxes` (una riga per persona rilevata):
`vertices` e `joints3d` sono metrici e includono già la traslazione stimata della
camera, `joints2d` è in pixel sull'immagine originale, e le rotazioni seguono la
convenzione di MHR, angoli di Eulero invece che axis-angle. Vedi
[predizione](/docs/predict) per le sorgenti, lo streaming e la gestione dei
risultati.

## Varianti

Due backbone dietro lo stesso modello corporeo MHR: `d3` usa un encoder DINOv3
ViT-H/16+, mentre `h` usa l'encoder ViT-H originale.

## Esportazione

<export-matrix />

L'esportazione della mesh corporea non è implementata: LibreYOLO non ha ancora
definito un contratto di grafo esportato per il task della mesh, compreso come
rappresentare il layout dei parametri di MHR fuori da PyTorch.

## Checkpoint

Tutti i file di pesi pubblicati per questa famiglia.

<checkpoint-table />

## Licenze

<provenance-box>

Il modello corporeo che i checkpoint pilotano, MHR (Momentum Human Rig), è una
release Meta separata sotto Apache-2.0. LibreYOLO scarica il suo asset TorchScript
dalla release pubblica di MHR stesso a runtime e lo mette in cache in locale;
quel file non è replicato da LibreYOLO e porta con sé i propri termini
Apache-2.0, non la SAM License.

</provenance-box>

## Citazione

<citation-block />

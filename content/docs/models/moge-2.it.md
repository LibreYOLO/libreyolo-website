---
title: MoGe-2
families:
  - moge2
seo_title: 'MoGe-2: predire, validare ed esportare le normali di superficie'
description: >-
  Usa MoGe-2 in LibreYOLO per la predizione densa delle normali di superficie.
  Installa, fai predizioni, valida ed esporta i checkpoint ufficiali ViT-S,
  ViT-B e ViT-L.
lead: >-
  MoGe-2 è un modello di geometria monoculare a singolo forward che predice un
  campo denso di normali di superficie a partire da una sola immagine RGB.
  LibreYOLO lo supporta solo per la stima delle normali, attraverso i checkpoint
  ufficiali ViT-S, ViT-B e ViT-L.
keywords:
  - MoGe-2
  - MoGe 2
  - surface normal estimation
  - normali di superficie
  - monocular geometry
  - normal map
  - mappa delle normali
  - dense prediction
  - DINOv2
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE, save=True)

        normal = result.normal_map
        print(normal.array.shape)   # vettori unitari (H, W, 3) float32
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMoGe2s-normal.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=518)

        print(metrics["metrics/mean_angular_error"])   # gradi
        print(metrics["metrics/median_angular_error"])
        print(metrics["metrics/within_11_25"])          # percentuale di pixel
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMoGe2s-normal.pt data=my-dataset.yaml imgsz=518
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        model.export(format="onnx", imgsz=518)
        model.export(format="tensorrt", imgsz=518, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreMoGe2s-normal.pt format=onnx imgsz=518

        libreyolo export model=LibreMoGe2s-normal.pt format=tensorrt imgsz=518
        half=True
    - label: Usare il file esportato
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.normal_map.array.shape)
source_hash: ddfacf6b7e9729f6
---

## Installazione

MoGe-2 non richiede nessun extra opzionale. Tutto ciò che importa è già nell'installazione base.

```bash
pip install libreyolo
```

## Predizione

I pesi vengono scaricati automaticamente al primo utilizzo: LibreYOLO preleva la
dimensione corrispondente direttamente dai checkpoint ufficiali e la mette in
cache in locale.

<code-tabs name="predict" />

MoGe-2 restituisce un campo denso invece di un insieme di rilevamenti, quindi
`result.boxes` è vuoto e `conf`, `iou` e `max_det` non hanno alcun effetto.
Il risultato sta in `result.normal_map`: un array `(H, W, 3)` di vettori unitari
nel sistema di riferimento della camera OpenCV, dove `+x` va a destra, `+y` va
verso il basso, `+z` entra nella scena, e una superficie rivolta verso la camera
vale `(0, 0, -1)`. Predire una lista di immagini esegue un forward pass per
immagine; questa famiglia non ha un percorso rapido a batch impilato. Vedi
[predizione](/docs/predict) per sorgenti, streaming e gestione dei risultati.

## Varianti

Tre dimensioni di encoder sono distribuite come checkpoint separati: ViT-S,
ViT-B e ViT-L, tutte alla stessa risoluzione di input. Il banco di benchmark di
LibreYOLO non ha misurato questa famiglia, quindi non ci sono numeri di
accuratezza pubblicati per confrontarle; scegli una dimensione in base al tuo
budget di calcolo.

## Validazione

`val()` misura l'errore angolare rispetto a un dataset appaiato di mappe di
normali: immagini affiancate a PNG di normali a 16 bit con lo stesso nome, più
una maschera di validità opzionale, così i pixel di padding e quelli non validi
non vengono mai conteggiati. Restituisce l'errore angolare medio e mediano in
gradi, più la percentuale di pixel entro 11.25, 22.5 e 30 gradi.

<code-tabs name="val" />

## Esportazione

<export-matrix />

L'esportazione delle normali usa un contratto di runtime a risoluzione fissa e
batch 1: `dynamic` e un `batch` diverso da 1 vengono rifiutati, e `imgsz` deve
essere divisibile per la dimensione delle patch dell'encoder ViT, cosa che
LibreYOLO verifica prima che l'esecuzione inizi. Un artefatto esportato si
ricarica con `LibreYOLO()` in base al suffisso del file, quindi un file `.onnx`
si comporta come un checkpoint e restituisce lo stesso `Results`.

<code-tabs name="export" />

## Licenze

<provenance-box>

LibreYOLO non copia questi checkpoint nella propria organizzazione.
`LibreYOLO("LibreMoGe2s-normal.pt")` scarica la dimensione corrispondente
direttamente dai repository ufficiali su Hugging Face a una revisione fissata, e
verifica il file rispetto a un checksum SHA-256 registrato prima dell'uso.

</provenance-box>

## Citazione

<citation-block />

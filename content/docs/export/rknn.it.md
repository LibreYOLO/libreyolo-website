---
title: RKNN
seo_title: "Esportare in RKNN per le NPU Rockchip"
description: "Compila un detector LibreYOLO in un artefatto Rockchip .rknn: l'SDK del produttore che installi tu, le quattro varianti RK3588 validate e la parità nel simulatore."
lead: "RKNN è il formato compilato per le NPU di Rockchip. LibreYOLO esporta un intermedio ONNX a opset 19, lo compila con l'SDK RKNN Toolkit2 e può confrontare il grafo compilato con ONNX Runtime nel simulatore host di Toolkit2, senza bisogno di una scheda."
keywords:
  - esportare yolo rknn
  - npu rockchip
  - rk3588
  - rknn-toolkit2
  - parità simulatore rknn
  - inferenza orange pi rockchip
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="rknn", name="rk3588")'
    mono: true
  - label: Scrive
    value: "Un file .rknn, un sidecar .rknn.metadata.json e un report .rknn.parity.json quando verify=True"
  - label: Extra
    value: "Nessuno su PyPI. rknn-toolkit2 è un SDK del produttore che installi tu."
  - label: Si ricarica con
    value: "Non tramite LibreYOLO. L'artefatto gira sulla scheda con il runtime di Rockchip."
  - label: Forme
    value: "Quadrata fissa, batch 1, opset 19. Tutti e tre i vincoli sono imposti."
  - label: Precisione
    value: "La build in virgola mobile del produttore. half=True e int8=True vengono rifiutati."
  - label: Ambito
    value: "Quattro varianti di rilevamento su RK3588: YOLO9-t, YOLO9-E2E-t, PicoDet-s e YOLO-NAS-s"
verification: "Letto da libreyolo/export/rknn.py, libreyolo/export/exporter.py, libreyolo/export/support.py e docs/rknn.md sul branch dev. I numeri di parità misurati provengono dal record di validazione datato 2026-08-04 in docs/rknn.md."
snippets:
  install:
    - label: Lato LibreYOLO
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: SDK del produttore, installato da te
      language: bash
      code: |
        # rknn-toolkit2 è un SDK Rockchip con licenza separata. LibreYOLO non lo
        # include né lo installa. Solo Linux x86_64; su Windows usa WSL2 o un
        # container Linux.
        #
        # Toolkit2 2.3.2 richiede setuptools<81 e fallisce con ONNX 1.19 o
        # successivi, che hanno rimosso onnx.mapping mentre il suo compilatore
        # continua a importarlo.
        pip install "setuptools==80.9.0" "onnx==1.18.0"

        # Poi installa la wheel rknn-toolkit2 corrispondente dal repository di
        # wheel di Rockchip, e verifica che si importi:
        python -c "import rknn.api; print('rknn-toolkit2 ready')"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Scrive weights/LibreYOLO9t.rknn e weights/LibreYOLO9t.rknn.metadata.json
        path = model.export(format="rknn", name="rk3588", imgsz=640, verify=True)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format rknn --name rk3588 \
          --imgsz 640 --verify
    - label: Argomenti
      language: python
      code: |
        model.export(
            format="rknn",
            name="rk3588",     # piattaforma target; funzionano anche target= e target_platform=
            imgsz=640,         # deve coincidere con il canvas registrato della variante
            batch=1,           # qualsiasi altro valore solleva NotImplementedError
            dynamic=False,     # True solleva ValueError
            opset=19,          # qualsiasi altro valore solleva NotImplementedError
            verify=False,      # True esegue il simulatore su PC e blocca in caso di parità insufficiente
        )
  parity:
    - label: Parità senza scheda su un artefatto ONNX esistente
      language: python
      code: |
        import numpy as np
        from libreyolo.export import verify_rknn_simulator_parity

        input_tensor = np.random.default_rng(0).standard_normal(
            (1, 3, 640, 640), dtype=np.float32
        )
        metrics = verify_rknn_simulator_parity(
            "weights/LibreYOLO9t.onnx",
            input_tensor,
            target_platform="rk3588",
            rtol=1e-3,
            atol=1e-4,
            raise_on_failure=False,
        )
        print(metrics)
  support:
    - label: Controllare una famiglia e un task prima di compilare
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Installazione

Per la compilazione serve RKNN Toolkit2 di Rockchip, distribuito come SDK del
produttore sotto la licenza di Rockchip e non è una dipendenza di LibreYOLO. Non
esiste un extra `libreyolo[rknn]`, e niente di questo formato si installa con una
sola riga.

<code-tabs name="install" />

Per compilare o per controllare la parità numerica non serve una scheda. Una
scheda RK3588 serve per le misure di latenza, consumo e comportamento termico,
nessuna delle quali è stata registrata.

## Esportazione

<code-tabs name="export" />

La richiesta viene validata contro un elenco di varianti di modello esatte prima
che venga compilato qualsiasi cosa, e viene validato anche il canvas: passare un
`imgsz` diverso da quello con cui la variante è stata registrata solleva un
errore invece di compilare in silenzio qualcosa che non è stato testato.
LibreYOLO scrive un intermedio ONNX a opset 19, lo compila, facoltativamente lo
simula, e poi rimuove l'intermedio.

I metadati stanno in un sidecar chiamato `<model>.rknn.metadata.json`, perché il
formato RKNN non ha un campo di metadati portabile.

`verify=True` esegue il simulatore su PC di Toolkit2 all'interno della stessa
sessione che ha compilato l'artefatto, confronta ogni output con ONNX Runtime
sullo stesso input, e scrive `<model>.rknn.parity.json` con le metriche di errore
per ciascun output. Le soglie sono una similarità del coseno di almeno 0.9999 e
un RMSE normalizzato di al massimo 0.02, applicate a qualsiasi output che non sia
già vicino elemento per elemento; la build in virgola mobile del produttore
abbassa i tensori interni a precisione half, quindi un `allclose` stretto non
regge nemmeno quando i box decodificati sono stabili. Un'esecuzione fallita
scrive `<model>.rknn.failed.parity.json`, scarta il candidato, e lascia intatta
qualsiasi esportazione precedente andata a buon fine a quel percorso.

Per confrontare un artefatto ONNX che hai già, senza esportare di nuovo:

<code-tabs name="parity" />

Il simulatore di Toolkit2 esegue il grafo in memoria prodotto da `load_onnx` e
`build`. Non può ricaricare un file `.rknn` specifico per un target senza una
scheda, ed è per questo che `verify=True` fa compilazione, esportazione e
simulazione in un'unica sessione.

## Eseguire l'artefatto

Non c'è una voce RKNN in `libreyolo/backends`, quindi `LibreYOLO()` non carica un
file `.rknn`. L'artefatto compilato viene messo in produzione sulla scheda ed
eseguito dal runtime di Rockchip, e lì il preprocessing, la decodifica, l'NMS e
il riscalamento delle coordinate sono responsabilità dell'applicazione.

`<model>.rknn.metadata.json` contiene i nomi delle classi, la dimensione di
input, il task e la piattaforma target, cioè quello che serve a un'applicazione
per riprodurre il postprocessing di LibreYOLO. Distribuiscilo insieme al modello
compilato.

Per un controllo lato host che non richiede la scheda, tieni un artefatto ONNX
alla stessa forma fissa e confrontalo nel simulatore, come sopra.

## Vincoli

Compilano quattro combinazioni, e sono varianti di modello più che famiglie:

| Variante | Task | Canvas | Target |
|---|---|---:|---|
| YOLO9-t | detect | 640 | RK3588 |
| YOLO9-E2E-t | detect | 640 | RK3588 |
| PicoDet-s | detect | 320 | RK3588 |
| YOLO-NAS-s | detect | 640 | RK3588 |

Tutto il resto viene rifiutato prima della compilazione, con il messaggio che in
questa versione RKNN è limitato alle esatte varianti di rilevamento testate nel
simulatore. Esistono risultati di sola compilazione per altri modelli, ma non
vengono deliberatamente presentati come supporto: nella stessa sessione di
misura, RF-DETR ha lasciato due nodi `GridSample` del decoder non abbassati, e
D-FINE, RT-DETR, RT-DETRv2, RT-DETRv4, DEIM, DEIMv2 ed EC hanno compilato e
simulato con output decodificati sostanzialmente sbagliati.

Batch 1, forme statiche, opset 19. `half=True` viene rifiutato, perché RKNN non
espone il contratto `half` di LibreYOLO, e `int8=True` viene rifiutato finché non
esistono risultati di calibrazione rappresentativa e di accuratezza sul task.

Gli altri target Rockchip vengono rifiutati: `rk3588` è l'unica piattaforma
validata.

Per la griglia completa di famiglie e task, vedi
[la matrice di esportazione](/docs/reference/export-matrix). Per una singola
combinazione:

<code-tabs name="support" />

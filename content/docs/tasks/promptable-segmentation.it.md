---
title: Segmentazione promptable
seo_title: Segmentazione promptable in LibreYOLO
description: >-
  Trasforma un punto, un box o un concetto testuale nella maschera di un oggetto
  con LibreYOLO. Carica SAM, SAM 2, SAM 3, EdgeTAM, MobileSAM o PicoSAM3 tramite
  LibreSAM.
lead: >-
  La segmentazione promptable trasforma un clic in una maschera: indichi un
  oggetto, o ci disegni intorno un box, e il modello ne restituisce il contorno.
  In LibreYOLO non è una chiave di task a sé, ma un tier di modelli, caricato
  tramite la factory LibreSAM, i cui risultati sono normali Results di
  segmentazione.
keywords:
  - promptable segmentation
  - segmentazione interattiva
  - segment anything python
  - prompt a punti SAM
  - prompt box SAM
  - SAM python
  - maschera da un clic
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompt a punti e a box
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # Un punto è [x, y] in pixel; in labels 1 è positivo, 0 negativo.
        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])
        print(result.masks.xy)      # poligoni
        print(result.boxes.xyxy)    # box aderenti derivati dalle maschere

        # Un prompt box produce una maschera per box.
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
    - label: 'Codifica una volta, fai molti prompt'
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # set_image esegue una volta sola il pesante encoder dell'immagine e ne
        # mette in cache il risultato.
        model.set_image(SAMPLE_IMAGE)
        first = model.predict(points=[640, 420], labels=[1])
        second = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
    - label: Segmentare tutto
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # Nessun prompt significa una griglia di punti su tutta l'immagine. La
        # griglia predefinita di 32 per lato è ~1024 passaggi del decoder,
        # lenti su CPU.
        result = model.predict(SAMPLE_IMAGE, points_per_side=8)
        print(len(result.masks))
    - label: Maschere di ambiguità
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # Un punto può indicare una manica, una camicia o una persona.
        # multimask=True restituisce tutte e tre le maschere intero-contro-parte
        # invece della migliore.
        result = model.predict(
            SAMPLE_IMAGE, points=[640, 420], labels=[1], multimask=True
        )
        print(len(result.masks))
source_hash: bb70ff24e6c0a767
---

## Definizione

La segmentazione promptable prende un'immagine più un prompt spaziale e
restituisce la maschera di ciò che il prompt indica. Non viene classificato
nulla: non c'è nessuna lista di classi, e `result.boxes` contiene box aderenti
derivati dalle maschere, non rilevamenti a sé stanti. `result.masks` porta i
dati delle maschere e `result.masks.xy` i loro poligoni.

Il prompt è l'interfaccia. `points` sono coordinate `[x, y]` in pixel, un
insieme per oggetto, con `labels` che marca ogni punto come positivo (1,
includi questo) o negativo (0, escludi questo). `bboxes` è
`[x1, y1, x2, y2]`, una maschera per box. Punti e box si possono combinare, e in
quel caso si accoppiano per oggetto e devono avere la stessa lunghezza. Omettere
ogni prompt esegue il percorso segment-everything, una griglia di punti
sull'immagine.

Un singolo punto è ambiguo per costruzione. Cliccare su una manica può voler
dire la manica, la camicia o la persona, quindi `multimask=True` restituisce per
ogni prompt tutte e tre le maschere intero-contro-parte invece della singola
migliore. `conf` filtra in base all'IoU predetto dal modello, un punteggio di
qualità della maschera, non una confidenza di rilevamento.

LibreYOLO non ha una chiave di task `promptable`. Il tier si registra come
`segment`, la stessa chiave che usa la segmentazione di istanze. A distinguerlo
è la forma della chiamata, ed è per questo che ha una factory propria,
`LibreSAM()`, sorella di `LibreYOLO()`, `LibreOpenVocab()` e `LibreVLM()`. Una
singola firma `predict(image)` non può esprimere il ciclo per cui questi modelli
sono costruiti: `set_image()` esegue una volta sola l'encoder dell'immagine e
mette in cache gli embedding, ogni chiamata successiva a `predict()` con
`source=None` paga solo la decodifica del prompt, e `reset_image()` svuota la
cache. L'encoder dell'immagine è il costo dominante e viene eseguito una volta
per immagine, quindi un secondo prompt sulla stessa immagine lo salta del tutto.

## Modelli

Sei famiglie si caricano tramite `LibreSAM` per alias.

[SAM](/docs/models/sam) è quella predefinita, nelle dimensioni `base`, `large` e
`huge`, scrivibili anche `b`, `l` e `h`.

[SAM 2](/docs/models/sam-2), come `sam2-tiny`, `sam2-small`, `sam2-base-plus` e
`sam2-large`. LibreYOLO ne supporta il percorso su immagine.

[SAM 3](/docs/models/sam-3), come `sam3`, è l'unica famiglia che accetta un
prompt di concetto testuale: `text="yellow school bus"` restituisce ogni istanza
corrispondente. Passare `text=` a qualunque altra famiglia solleva un'eccezione
con un messaggio che nomina SAM 3. I suoi pesi arrivano da Meta sotto la SAM
License personalizzata invece che sotto la licenza MIT di LibreYOLO, e il
repository è ad accesso controllato: accetta i termini sulla pagina del modello
e autenticati con `hf auth login` prima del primo download. Leggi
[SAM 3](/docs/models/sam-3) prima di metterlo in produzione.

[EdgeTAM](/docs/models/edgetam), come `edgetam`, è una variante on-device di
SAM 2. LibreYOLO ne supporta il percorso su immagine.

[MobileSAM](/docs/models/mobilesam), come `mobilesam`, sostituisce l'encoder
ViT-H di SAM con uno TinyViT distillato.

[PicoSAM3](/docs/models/picosam3), come `picosam3`, è una CNN compatta per
regioni indicate da un box su sensori edge. Qui i prompt box sono tutto il
contratto: punti, testo, maschera, multimask e segment-everything sollevano
tutti un'eccezione con un messaggio che rimanda a SAM 2 o SAM 3.

L'extra del tier copre le quattro famiglie che si caricano tramite
`transformers`:

```bash
pip install "libreyolo[sam]"
```

MobileSAM e PicoSAM3 sono port nativi di LibreYOLO e non hanno bisogno di
installare `transformers` per funzionare.

## Predizione

<code-tabs name="predict" />

`source` e `set_image()` sono alternative, non una sequenza: passa un'immagine a
`predict()` per una chiamata singola, oppure chiama prima `set_image()` e poi
`predict(source=None)` per ogni prompt. Passare `device=` a `predict()` sposta il
modello per quella chiamata e per tutte le successive, e invalida gli embedding
messi in cache.

Segment-everything è la modalità costosa. `points_per_side` vale 32 per default,
cioè circa 1024 passaggi del decoder sull'immagine; abbassalo per qualsiasi cosa
interattiva su CPU. In quella modalità `conf`, se lasciato non impostato, applica
la soglia di griglia della famiglia, mentre nel percorso con prompt un `conf` non
impostato tiene tutte le maschere. Passa `conf=0.0` per disattivare il
filtraggio in entrambe le modalità, e `max_det` per limitare quante maschere
tornano indietro.

I prompt a maschera non sono supportati in questa versione, e `masks=` solleva
un'eccezione invece di essere ignorato. Anche `track()` solleva un'eccezione in
tutto il tier: questi sono segmentatori di immagini, quindi esegui `predict()`
per ogni frame. Vedi [predizione](/docs/predict) per le sorgenti e la gestione
dei risultati.

## Addestramento

Nessuna famiglia di questo tier si addestra dentro LibreYOLO. `train()` solleva
un'eccezione: fai fine-tuning a monte e carica i pesi risultanti.

## Validazione

Non esiste un validatore per questo tier, e `val()` solleva un'eccezione. Una
maschera promptable non ha un insieme fisso di classi su cui calcolare un
punteggio, quindi le solite metriche di rilevamento e segmentazione non hanno
nulla a cui agganciarsi. Valutare una maschera ottenuta da un prompt significa
confrontarla con una maschera di riferimento che fornisci tu, sui prompt che ti
interessano.

## Esportazione

L'esportazione è fuori dallo scopo del tier nel suo insieme e `export()` solleva
un'eccezione, tranne in un caso. [PicoSAM3](/docs/models/picosam3) esporta in
ONNX la sua CNN di regione grezza 96x96 come `roi_image -> mask_logits`; il
ritaglio del box e il ridimensionamento della maschera verso le coordinate
dell'immagine restano in Python. Ogni altra famiglia gira tramite `predict()` in
PyTorch. Vedi [esportazione](/docs/export) per i formati disponibili altrove
nella libreria.

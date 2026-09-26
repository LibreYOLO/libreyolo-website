---
title: Risoluzione dei problemi
seo_title: Correggere gli errori più comuni di LibreYOLO
description: >-
  Gli errori che LibreYOLO solleva più spesso, che cosa significa ciascuno e
  come si risolve. Include due malfunzionamenti che producono un risultato
  sbagliato invece di sollevare un'eccezione.
lead: >-
  Errori raggruppati per il messaggio che vedi. Le ultime due voci coprono il
  problema opposto: codice che gira, restituisce qualcosa di plausibile ed è
  sbagliato.
keywords:
  - errore libreyolo
  - modulenotfounderror libreyolo
  - libreyolo cuda out of memory
  - libreyolo notimplementederror
  - risolvere errori libreyolo
last_verified: 1.5.0
source_hash: e271ab29b789865a
---

Gli errori sono raggruppati per il testo che vedi. Se il tuo messaggio non è
qui, le [FAQ](/docs/faq) rispondono alle domande che non riguardano
malfunzionamenti, e `libreyolo models` riporta che cosa la tua installazione
riesce davvero a caricare.

## ModuleNotFoundError che nomina un pacchetto che non hai mai importato

Alcune famiglie richiedono un extra opzionale. Il messaggio nomina il pacchetto
mancante invece dell'extra, quindi la soluzione non è sempre ovvia dal
traceback.

Esegui `libreyolo models`. Ogni famiglia a cui manca una dipendenza viene
stampata con il comando pip esatto che l'abilita, così non devi risalire tu
stesso dal pacchetto all'extra. `libreyolo models --json` stampa le stesse
informazioni come oggetto.

La [pagina di installazione](/docs/install) elenca ogni extra e che cosa copre.

## ONNX inference requires onnxruntime

```
ImportError: ONNX inference requires onnxruntime. Install with: pip install onnxruntime
```

Il pacchetto base non dipende da un runtime, perché quale ti serve dipende dal
tuo hardware. Installa `onnxruntime` per la CPU oppure `onnxruntime-gpu` per
CUDA. Entrambi forniscono lo stesso modulo `onnxruntime`, quindi installane
uno, non tutti e due.

## ONNX model not found

```
FileNotFoundError: ONNX model not found: <path>
```

Il percorso viene risolto rispetto alla directory di lavoro, non allo script.
Compare anche quando un'esportazione ha scritto in silenzio altrove:
`export()` restituisce il percorso in cui ha scritto, quindi cattura il valore
di ritorno invece di dare per scontato un nome.

## NotImplementedError da train()

Non tutte le famiglie si addestrano. Alcune sono state portate solo per
predizione, validazione ed esportazione, e il loro `train()` solleva
un'eccezione invece di fingere di funzionare.

La [voce delle FAQ](/docs/faq) spiega il motivo. Per verificare una famiglia
specifica prima di scrivere uno script di addestramento, la pagina del modello
indica se è addestrabile.

## NotImplementedError da export()

Una famiglia può supportare un task e comunque non esportarlo. EoMT è il caso
in cui ci si imbatte più spesso: `export()` accetta il task semantico e solleva
un'eccezione per `segment` e `panoptic`, perché il contratto di runtime
query-mask che servirebbe non è definito.

```
NotImplementedError: LibreEoMT instance and panoptic export need query-mask runtime contracts.
```

La pagina di ogni famiglia riporta una matrice di esportazione che mostra quali
combinazioni di task e formato sono validate.

## CUDA out of memory

Riduci prima `batch`, poi `imgsz`. Entrambi cambiano la memoria più o meno in
proporzione alla loro dimensione, ma il batch è quello che puoi abbassare senza
cambiare quello che il modello vede.

Se fallisce in validazione anziché in addestramento, la validazione usa una
propria dimensione del batch, quindi abbassa anche quella.

Su Windows, una GPU che pilota il display ha un secondo tipo di
malfunzionamento che sembra un errore CUDA casuale invece di un esaurimento
della memoria: il driver resetta una GPU che smette di rispondere più a lungo
del timeout, terminando qualunque cosa fosse in esecuzione. I kernel lunghi
sulla scheda che pilota il tuo monitor possono farlo scattare.

## I pesi non si scaricano

I pesi vengono scaricati da Hugging Face al primo utilizzo e messi in cache in
locale. Le [FAQ](/docs/faq) spiegano dove si trova la cache e come lavorare
completamente offline.

Se un download restituisce 404, controlla il nome del file che hai passato.
L'URL viene derivato dal nome, suffisso del task incluso, quindi un nome che non
corrisponde a un checkpoint pubblicato produce un URL che non esiste. La
tabella dei checkpoint su ogni pagina di modello elenca i nomi esatti dei file
pubblicati.

## L'addestramento si blocca o riparte su Windows

Windows non ha `fork`, quindi i worker del dataloader partono reimportando il
tuo script. Senza una guardia `if __name__ == "__main__":`, ogni worker riesegue
la tua chiamata di addestramento, il che porta a un deadlock oppure genera
processi all'infinito.

```python
def main():
    ...  # costruisci il modello e chiama train()

if __name__ == "__main__":
    main()
```

Anche impostare `workers=0` lo evita, a costo del throughput. La guardia è la
soluzione migliore.

## Due malfunzionamenti che non sollevano eccezioni

Il resto di questa pagina parla di errori. Questi due sono peggio, perché il
codice gira e restituisce qualcosa che sembra giusto.

### Indicizzare un singolo risultato

`predict()` restituisce un solo `Results` per un'immagine, e una lista per
più immagini. Indicizzare il valore restituito per una singola immagine
seleziona un *rilevamento*, non un'immagine:

```python
result = model.predict("image.jpg")   # un Results
result.boxes                          # ogni rilevamento, corretto
result[0].boxes                       # UN solo rilevamento, in silenzio
```

Non viene sollevata alcuna eccezione, perché indicizzare un `Results` è
un'operazione valida che restituisce un sottoinsieme. Il codice scritto per la
forma a lista riporta silenziosamente un solo box per immagine. Indicizza solo
ciò che sai essere una lista.

### Leggere le metriche come attributi

`val()` restituisce un semplice dizionario le cui chiavi sono i nomi delle
metriche, non un oggetto con accesso per attributi:

```python
metrics = model.val(data="coco8.yaml")
metrics["metrics/mAP50-95"]   # corretto
metrics.box.map               # AttributeError
```

Le chiavi hanno i prefissi `metrics/` e `speed/`. Stampa il dizionario una
volta per vedere che cosa ha prodotto il tuo task, dato che l'insieme cambia da
task a task.

## Controllare un dataset prima di addestrare

La maggior parte degli errori in addestramento dipende dal dataset.
`libreyolo doctor data.yaml` esegue controlli di integrità su un dataset di
rilevamento e riporta i risultati per gravità, il che è più rapido che leggere
un traceback alla prima epoca.

```python
from libreyolo import doctor

report = doctor.diagnose("data.yaml", imgsz=640)
if report.errors:
    ...
```

Vedi il [comando doctor](/docs/cli/doctor) per il catalogo dei controlli.

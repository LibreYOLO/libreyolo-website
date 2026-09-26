---
title: Fehlerbehebung
seo_title: Häufige LibreYOLO-Fehler beheben
description: >-
  Die häufigsten LibreYOLO-Fehler, ihre Bedeutung und die jeweilige Lösung.
  Einschließlich zweier Fehler, die falsche Ausgaben erzeugen, statt eine
  Ausnahme auszulösen.
lead: >-
  Die Fehler sind nach der angezeigten Meldung gruppiert. Die letzten beiden
  Einträge behandeln das umgekehrte Problem: Code, der läuft, ein plausibles
  Ergebnis zurückgibt und dennoch falsch ist.
keywords:
  - libreyolo fehler
  - modulenotfounderror libreyolo
  - libreyolo cuda out of memory
  - libreyolo notimplementederror
  - libreyolo fehlerbehebung
last_verified: 1.5.0
source_hash: e271ab29b789865a
---

Die Fehler sind nach dem angezeigten Text gruppiert. Wenn deine Meldung hier
nicht aufgeführt ist, beantwortet die [FAQ](/docs/faq) Fragen, bei denen es
nicht um Fehler geht. `libreyolo models` zeigt, was deine Installation
tatsächlich laden kann.

## ModuleNotFoundError nennt ein nie importiertes Paket

Einige Familien benötigen ein optionales Extra. Die Meldung nennt das fehlende
Paket und nicht das Extra. Die Lösung ist daher aus dem Traceback nicht immer
offensichtlich.

Führe `libreyolo models` aus. Jede Familie mit fehlender Abhängigkeit wird
zusammen mit dem genauen pip-Befehl ausgegeben, der sie aktiviert. Du musst das
Paket daher nicht selbst dem Extra zuordnen. `libreyolo models --json` gibt
dieselben Daten als Objekt aus.

Die [Installationsseite](/docs/install) führt alle Extras und ihren Umfang auf.

## ONNX-Inferenz benötigt onnxruntime

```
ImportError: ONNX inference requires onnxruntime. Install with: pip install onnxruntime
```

Das Basispaket hängt nicht von einer Runtime ab, weil die richtige Auswahl von
deiner Hardware abhängt. Installiere `onnxruntime` für die CPU oder
`onnxruntime-gpu` für CUDA. Beide stellen dasselbe Modul `onnxruntime` bereit.
Installiere daher nur eines von beiden.

## ONNX-Modell nicht gefunden

```
FileNotFoundError: ONNX model not found: <path>
```

Der Pfad wird relativ zum Arbeitsverzeichnis und nicht relativ zum Skript
aufgelöst. Die Meldung erscheint auch, wenn ein Export unbemerkt an eine andere
Stelle geschrieben wurde. `export()` gibt den geschriebenen Pfad zurück.
Speichere daher den Rückgabewert, statt einen Namen anzunehmen.

## NotImplementedError von train()

Nicht jede Familie kann trainiert werden. Einige sind nur für Vorhersage,
Validierung und Export portiert. Ihre Methode `train()` löst einen Fehler aus,
statt einen vermeintlichen Lauf zu starten.

Der [FAQ-Eintrag](/docs/faq) erläutert die Gründe. Auf der Modellseite einer
Familie kannst du vor dem Schreiben eines Trainingsskripts prüfen, ob sie
Training unterstützt.

## NotImplementedError von export()

Eine Familie kann eine Aufgabe unterstützen, ohne sie exportieren zu können.
EoMT ist ein häufig auftretender Fall. `export()` akzeptiert die semantische
Aufgabe und löst für `segment` und `panoptic` einen Fehler aus, weil der dafür
erforderliche Query-Masken-Runtime-Vertrag nicht definiert ist.

```
NotImplementedError: LibreEoMT instance and panoptic export need query-mask runtime contracts.
```

Die Seite jeder Familie enthält eine Exportmatrix mit den validierten
Kombinationen aus Aufgabe und Format.

## CUDA out of memory

Verringere zuerst `batch` und anschließend `imgsz`. Beide beeinflussen den
Speicherbedarf ungefähr proportional zu ihrer Größe. Nur die Batch-Größe kannst
du jedoch reduzieren, ohne zu verändern, was das Modell sieht.

Wenn der Fehler bei der Validierung statt beim Training auftritt, verwendet
die Validierung eine eigene Batch-Größe. Verringere auch diese.

Unter Windows gibt es bei einer Anzeige-GPU einen zweiten Fehlermodus, der wie
ein zufälliger CUDA-Fehler statt wie ein Speichermangel aussieht. Der Treiber
setzt eine GPU zurück, die länger als das Timeout nicht reagiert, und beendet
dadurch den laufenden Prozess. Lange Kernel auf der Grafikkarte, die deinen
Monitor ansteuert, können dieses Verhalten auslösen.

## Gewichte werden nicht heruntergeladen

Gewichte werden bei der ersten Verwendung von Hugging Face abgerufen und lokal
zwischengespeichert. Die [FAQ](/docs/faq) beschreibt den Speicherort des Caches
und den vollständig netzwerkfreien Betrieb.

Wenn ein Download mit 404 fehlschlägt, prüfe den übergebenen Dateinamen. Die
URL wird einschließlich Aufgabensuffix daraus abgeleitet. Ein Name, der keinem
veröffentlichten Checkpoint entspricht, erzeugt eine nicht vorhandene URL. Die
Checkpoint-Tabelle jeder Modellseite führt die genauen veröffentlichten
Dateinamen auf.

## Training hängt oder startet unter Windows neu

Windows unterstützt kein `fork`. Dataloader-Worker starten daher, indem sie
dein Skript erneut importieren. Ohne einen Guard
`if __name__ == "__main__":` führt jeder Worker deinen Trainingsaufruf erneut
aus. Dadurch entsteht entweder ein Deadlock oder es werden endlos neue Prozesse
gestartet.

```python
def main():
    ...  # Modell erstellen und train() aufrufen

if __name__ == "__main__":
    main()
```

Mit `workers=0` lässt sich das Problem ebenfalls vermeiden, allerdings auf
Kosten des Durchsatzes. Der Guard ist die bessere Lösung.

## Zwei Fehler ohne Ausnahme

Der Rest dieser Seite behandelt Fehlermeldungen. Die folgenden beiden Fälle
sind schwerwiegender, weil der Code läuft und ein scheinbar richtiges Ergebnis
zurückgibt.

### Indizieren eines einzelnen Ergebnisses

`predict()` gibt für ein Bild ein `Results`-Objekt und für mehrere Bilder eine
Liste zurück. Wenn du die Rückgabe für ein einzelnes Bild indizierst, wählst du
eine *Erkennung* und kein Bild aus:

```python
result = model.predict("image.jpg")   # ein Results-Objekt
result.boxes                          # alle Erkennungen, richtig
result[0].boxes                       # EINE Erkennung, unbemerkt
```

Es wird keine Ausnahme ausgelöst, weil das Indizieren eines `Results`-Objekts
eine gültige Operation ist, die eine Teilmenge zurückgibt. Code, der für die
Listenform geschrieben wurde, meldet so unbemerkt eine Box pro Bild. Indiziere
nur Werte, von denen du weißt, dass sie Listen sind.

### Metriken als Attribute lesen

`val()` gibt ein einfaches Dictionary mit Metriknamen als Schlüsseln zurück und
kein Objekt mit Attributzugriff:

```python
metrics = model.val(data="coco8.yaml")
metrics["metrics/mAP50-95"]   # richtig
metrics.box.map               # AttributeError
```

Die Schlüssel besitzen die Namensräume `metrics/` und `speed/`. Gib das
Dictionary einmal aus, um zu sehen, was deine Aufgabe erzeugt hat, da sich die
Schlüssel je nach Aufgabe unterscheiden.

## Datensatz vor dem Training prüfen

Die meisten Trainingsfehler entstehen durch Datensatzprobleme.
`libreyolo doctor data.yaml` führt Integritätsprüfungen für einen
Erkennungsdatensatz aus und meldet Funde nach Schweregrad. Dies ist schneller,
als den Traceback der ersten Epoche zu untersuchen.

```python
from libreyolo import doctor

report = doctor.diagnose("data.yaml", imgsz=640)
if report.errors:
    ...
```

Den Prüfkatalog findest du beim [doctor-Befehl](/docs/cli/doctor).


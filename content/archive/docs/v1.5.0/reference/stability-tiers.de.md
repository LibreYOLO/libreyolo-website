---
title: Stabilitätsstufen
seo_title: Bedeutung der LibreYOLO-Support-Stufen
description: >-
  Das Stufenvokabular von LibreYOLO: drei Export-Support-Stufen, vier
  API-Stufen, sechs Abdeckungsgruppen und die Grenzen ihrer Aussagen.
lead: >-
  LibreYOLO verwendet das Wort Stufe für drei getrennte Konzepte: die Belege
  hinter einem Exportpfad, den Aufrufvertrag einer Modellfamilie und ihre
  Abdeckungsgruppe. Diese Seite definiert jedes Konzept und erklärt, was sich
  daraus nicht ableiten lässt.
keywords:
  - libreyolo support-stufe
  - validated available blocked
  - export-support-stufen
  - libreyolo abdeckungsgruppen
  - g0 g1 g2 g3 g4
  - modellstufen
last_verified: 1.5.0
verification: >-
  Exportstufen aus docs/adr/0011-export-support-tiers.md und
  libreyolo/export/support.py; Abdeckungsgruppen und Familienanzahlen aus
  MODEL_GROUPS in libreyolo/models/registry.py; From-Scratch-Sperre aus
  libreyolo/models/base/model.py und libreyolo/cli/commands/train.py;
  CLI-Inventar aus libreyolo/models/inventory.py; API-Stufen aus
  Paket-Docstrings und base.py-Verträgen unter libreyolo/models/sam/, openvocab/
  und vlm/, jeweils für v1.5.0. Leserorientierte Gruppenlabels (Flagship, Core,
  Supported, Inference only, Museum, Sibling tier) sind die eigene
  Website-Terminologie für dieselben Gruppen aus src/data/docs/registry.json.
snippets:
  usage:
    - label: Beide Klassifizierungen einer Familie lesen
      language: python
      code: |
        from libreyolo.models.registry import GROUPS, group_of
        from libreyolo.export.support import get_support, validated_alternatives

        family = "yolo9"

        group = group_of(family)
        print(group, GROUPS[group])

        print(get_support(family, "detect", "onnx").tier)
        print(validated_alternatives(family, "detect"))
source_hash: de545894b0d125e4
---

## Export-Support-Stufen

Diese Stufe bestimmt, ob ein Aufruf gelingt. Sie gilt für das Tripel
`(family, task, format)`. Jede Kombination besitzt genau eine Stufe.

| Stufe | Bedeutung | Verhalten von `export()` |
|---|---|---|
| `validated` | Numerische Parität wird im CI-System oder einem dokumentierten nächtlichen Lauf geprüft | Wird ausgeführt |
| `available` | Konvertierung ist implementiert, aber Belege für numerische Runtime-Parität wurden noch nicht aufgezeichnet | Wird ausgeführt |
| `blocked` | Kein unterstützter Pfad | Löst in der Vorprüfung einen `NotImplementedError` mit Begründung aus |

Sowohl „validated“ als auch „available“ werden ohne Bestätigung oder pauschale
Warnung ausgeführt. Der Unterschied liegt in den Belegen und nicht in der
Erlaubnis. Hinter einem validierten Eintrag stehen ein Paritätstest und ein
`since`-Release, hinter einem verfügbaren noch nicht. Eine CoreML-Konvertierung
ohne Vorhersagelauf unter macOS ist zum Beispiel verfügbar, aber nicht validiert.

Eine blockierte Kombination scheitert vor Abhängigkeitsprüfungen,
Kalibrierungsladen, Tracing oder Artefakterstellung. Es wird daher nichts
Unvollständiges geschrieben.

Jede validierte Zelle enthält eine Einschränkung mit der Konfiguration, aus der
der Paritätswert stammt. Normalerweise sind dies eine feste Eingabe-Canvas,
Batch-Größe 1, FP32 und eine benannte Runtime-Version. Betrachte sie als Aussage
über diese Konfiguration und nicht über das Format insgesamt. Die Regeln für
Zellen ohne expliziten Eintrag stehen auf der Seite zur
[Exportmatrix](/docs/reference/export-matrix).

<code-tabs name="usage" />

## API-Stufen

Diese Stufe bestimmt die Form eines Aufrufs. Eine Familie gehört genau einer
Stufe an, die anhand des Aufrufvertrags und nicht der Architektur gewählt wird.

| Stufe | Factory | Vertrag |
|---|---|---|
| Detektor-Factory | `LibreYOLO` | Ein promptloser Forward Pass gibt jedes gefundene Objekt mit kalibrierten Scores zurück. Mitglieder registrieren sich durch Erkennung eines Checkpoints |
| Promptbasierte Segmentierung | `LibreSAM` | Ein Forward Pass ist ohne einen beim Aufruf bereitgestellten räumlichen oder Konzept-Prompt pro Bild bedeutungslos. Interaktiv und zustandsbehaftet: einmal encodieren, mehrfach prompten |
| Open-Vocabulary-Erkennung | `LibreOpenVocab` | Textkonditionierte diskriminative Detektoren. Die Klassenliste ist ein mit `set_classes` festgelegter Prompt |
| Vision-Language | `LibreVLM` | Ein generatives Modell, das als Detektor gesteuert wird. Die Klassenliste ist ein Prompt, die Confidence ein Platzhalter |

Die drei benachbarten Stufen registrieren sich bewusst nicht in der
Detektor-Factory. Deshalb kann `LibreYOLO("some-alias")` sie nicht erreichen.
Sie werden über Größenaliasse geladen und automatisch heruntergeladen, statt
durch Untersuchung eines Checkpoints.

Alle vier geben denselben Typ `Results` zurück. Nachgelagerter Code bleibt
daher über die Stufen hinweg unverändert. Die verfügbaren Methoden unterscheiden
sich. Die benachbarten Stufen lösen für `train()`, `val()` und `export()` einen
`NotImplementedError` aus. Die SAM- und Open-Vocabulary-Stufen tun dies auch
für `track()`. Jede Stufenseite führt ihre eigenen Ausschlüsse auf.

## Abdeckungsgruppen

Diese Klassifizierung bestimmt, welche Familien ein familienübergreifender
Testlauf umfasst. Leser sehen sie am häufigsten auf einer Modellseite. Jede
registrierte Familie gehört genau einer Gruppe an. Ein Test schlägt fehl, wenn
eine registrierte Familie keiner Gruppe zugeordnet ist. `GROUPS` in
`libreyolo/models/registry.py` ist die Quelle der Spalte „Bedeutung“.
`MODEL_GROUPS` in derselben Datei ordnet jede Familie zu, und die Spalte
„Familien“ zählt diese Zuordnungen direkt. Die Spalte „Label“ enthält den
kürzeren Namen, den die Website für dieselbe Gruppe im Kopf einer Modellseite
verwendet.

| Gruppe | Label | Familien | Bedeutung |
|---|---|---:|---|
| `g0` | Flagship | 2 | Flagship-Anker, die in der Abdeckung gemeinsamer Funktionen erforderlich sind |
| `g1` | Core | 10 | Abdeckungssatz trainierbarer Detektoren |
| `g2` | Supported | 14 | Zusätzlicher Abdeckungssatz trainierbarer Familien |
| `g3` | Inference only | 35 | Familien ohne Trainingsimplementierung |
| `g4` | Museum | 5 | Historische Familien mit Inferenzabdeckung |
| `s` | Sibling tier | 21 | Benachbarte APIs (SAM, Open-Vocabulary, VLM, Zero-Shot), getrennt abgedeckt |

Das sind 87 Familien in sechs Gruppen. Allein `g3` enthält mehr Familien als
alle anderen Gruppen zusammen, weil der Großteil des Registers aus
Inferenzlinien und Museum-Abdeckung statt aktiv trainierten Detektoren besteht.

Bei der Modellauswahl beschreibt die Gruppe die zu erwartende technische
Betreuung und nicht die Accuracy einer Familie. In `g0` und `g1` wird eine neue
Funktion entwickelt und zuerst implementiert. `g2` wird im CI-System grün
gehalten, erhält eine Funktion aber nach Gelegenheit und nicht zwingend in
derselben Release-Welle. `g3` beschreibt eine fehlende Funktion und keine
allgemeine Einschränkung. Vorhersage, Validierung und, sofern von der Familie
unterstützt, Export funktionieren weiterhin. `train()` einer Familie aus `g3`
oder `g4` löst einen `NotImplementedError` mit Begründung aus, statt
stillschweigend einen unvollständigen Vorgang auszuführen. Familien aus `s`
nehmen an diesem Kompromiss nicht teil, weil sie über ihre eigene Factory statt
`LibreYOLO()` geladen werden. Unter [Grundkonzepte](/docs/concepts) erfährst du,
wie eine Gruppe beim Lesen eines Checkpoint-Dateinamens neben Aufgabe, Familie
und Größe einzuordnen ist.

Eine Gruppe gewährt oder beschränkt keine nutzerseitige Funktion. Der Support
ergibt sich aus der implementierten API der Familie und formatspezifischen
Funktionsprüfungen, nie allein aus der Gruppenzugehörigkeit. Gruppen
klassifizieren Familien und keine Aufgaben. Ein aufgabenbezogener
Abdeckungslauf nennt die Aufgabe daher ausdrücklich, zum Beispiel „g1 detect“.

An zwei Stellen wird die Gruppe zur Laufzeit und nicht nur in Tests gelesen.
`collect_model_inventory()` in `libreyolo/models/inventory.py` ergänzt jeden
vom CLI-Inventar ausgegebenen Eintrag um die Gruppe. `pretrained=False` löst
den besonderen Pfad zur Neuinitialisierung von Grund auf nur für Familien aus
`g0` und `g1` aus. Außerhalb dieser beiden Gruppen wird die Prüfung in
`libreyolo/models/base/model.py` vollständig übersprungen. `pretrained=False`
erreicht dort die familieneigene Methode `train()` als gewöhnliches Keyword.

## Training

Eine Familie in `g3` oder `g4` besitzt keine Trainingsimplementierung. Ein
Aufruf von `train()` löst einen Fehler aus. Dies ist eine Eigenschaft des
Familiencodes und keine Wirkung der Gruppe. Die Gruppe dokumentiert den
Umstand lediglich.

Bei einer trainierbaren Familie ist es eine getrennte Frage, ob ein bestimmter
Augmentierungsparameter die Pipeline erreicht. Dafür gilt ein eigenes
dreiwertiges Vokabular: `used`, `gated_by_mosaic` und `ignored`. Siehe
[Augmentierungsmatrix](/docs/reference/augmentation-matrix).

## Grenzen einer Stufe

Eine Stufe ist keine Aussage zur Accuracy. Ein validierter Export besagt, dass
das Artefakt das native Modell innerhalb eines angegebenen Schwellenwerts
reproduziert. Daraus lässt sich nicht ableiten, wie gut das native Modell auf
einem Datensatz abschneidet. Benchmark-Werte stehen auf den Modellseiten.

Eine Stufe ist auch keine Lizenzaussage. Gewichtslizenzen unterscheiden sich
innerhalb einer Familie, und maßgeblich ist das Repository des konkreten
Checkpoints. Die Zugehörigkeit einer Familie zur Detektor-Factory sagt nichts
darüber aus, ob ihre veröffentlichten Gewichte kommerziell genutzt werden
dürfen.


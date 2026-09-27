---
title: Kernel
seo_title: LibreYOLO-Kernel-Registry und Hub-Kernel
description: >-
  Wie LibreYOLO beschleunigte Implementierungen auswählt: die Kernel-Registry
  unter libreyolo/kernels, der optionale MS-Deform-Attention-Kernel vom Hugging
  Face Hub und der Schalter für fusionierte Attention.
lead: >-
  Jede beschleunigte Operation in LibreYOLO besitzt eine portable
  Standardimplementierung und manchmal eine darüber registrierte schnellere
  Variante. Die Auswahl erfolgt zur Laufzeit anhand eines Prädikats. Eine
  fehlende optionale Abhängigkeit führt zu einem Fallback statt zu einem Fehler
  und ein exportierter Graph verwendet immer den portablen Pfad.
keywords:
  - LibreYOLO Kernel
  - LIBREYOLO_KERNELS
  - LIBREYOLO_HUB_KERNELS
  - Hub Kernels Extra
  - ms_deform_attn Kernel
  - set_fused_attention
  - LibreYOLO Triton Kernel
last_verified: 1.5.0
verification: >-
  Registry-API aus libreyolo/kernels/__init__.py in v1.5.0 gelesen,
  Attention-API aus libreyolo/kernels/attention/__init__.py und sdpa.py,
  Hub-Provider aus libreyolo/kernels/attention/ms_deform_attn.py einschließlich
  festgeschriebener Revision und Eignungsprädikat. Verzeichnisstruktur aus
  libreyolo/kernels/ aufgelistet. Extra-Definition aus pyproject.toml.
  Verhaltenshinweise und Benchmark-Werte aus docs/kernels.md. Entwicklung der
  Aktivierungsbedingung in v1.4.0 aus dem Commit zur RF-DETR-Slot-Verdrahtung
  und dem CHANGELOG-Eintrag für 1.5.0.
meta:
  - label: Paket
    value: libreyolo.kernels
    mono: true
  - label: Optionales Extra
    value: 'libreyolo[hub-kernels]'
    mono: true
  - label: Referenz erzwingen
    value: LIBREYOLO_KERNELS=off
    mono: true
snippets:
  usage:
    - label: Ausgewählte Implementierungen anzeigen
      language: python
      code: >
        import libreyolo.kernels as kernels


        # Zuordnung des Operations-Slots zum gewählten Implementierungsnamen
        oder "unavailable".

        print(kernels.active())
    - label: Referenzpfad erzwingen
      language: bash
      code: |
        # off und reference bedeuten dasselbe und überspringen außerdem
        # vollständig den Import beschleunigter Provider.
        LIBREYOLO_KERNELS=off python train.py
    - label: Hub-Kernel ohne Deinstallation deaktivieren
      language: bash
      code: |
        LIBREYOLO_HUB_KERNELS=0 python predict.py
    - label: Eine Familie auf fusionierte Attention umstellen
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.kernels.attention import set_fused_attention

        model = LibreYOLO("LibreSwinIRs.pt")

        # Gibt zurück, wie viele Attention-Module umgestellt wurden.
        print(set_fused_attention(model))
    - label: Eigene Implementierung registrieren
      language: python
      code: |
        import libreyolo.kernels as kernels

        kernels.register(
            "fake_quant_fp8",
            my_impl,
            name="mybackend",
            predicate=my_check,
        )
source_hash: 23d504e88b7959f8
---

## Registry

`libreyolo/kernels/` ist eine kleine Laufzeit-Registry für austauschbare Implementierungen. Ein Operations-Slot besitzt einen Namen wie `fake_quant_fp8` oder `ms_deform_attn`. Aufrufer fragen einen Slot bei der Registry ab und erhalten die erste registrierte Implementierung, deren Prädikat erfüllt ist. Die neueste Registrierung gewinnt. Trifft keine zu, wird auf die Referenzimplementierung zurückgegriffen.

Diese Struktur sorgt dafür, dass eine optionale Abhängigkeit nie zwingend erforderlich ist. Ein Rechner ohne Triton, CUDA oder das Paket `kernels` führt denselben Code aus und erzeugt dieselben Zahlen, lediglich langsamer.

| Funktion | Zweck |
|---|---|
| `active()` | Zuordnung des Operations-Slots zum ausgewählten Implementierungsnamen oder `"unavailable"` |
| `resolve(op)` | Aufrufbare Funktion, die ausgeführt würde, oder `None` |
| `register(op, impl, *, name, predicate=None)` | Implementierung an erster Stelle hinzufügen |
| `unregister(op, name)` | Implementierung entfernen |
| `clear_cache()` | Zwischengespeicherte Auflösung verwerfen |

<code-tabs name="usage" />

Löst ein Prädikat einen Fehler aus, wird dieser abgefangen und eine Warnung ausgegeben. Er wird nie weitergereicht. Eine fehlerhafte Drittanbieterimplementierung fällt dadurch auf den portablen Pfad zurück, statt die Vorhersage abzubrechen.

### Struktur

Der Baum ist zuerst nach Zweck und dann nach Backend organisiert. Ein Slot wird daher anhand seiner Berechnung gefunden und nicht anhand der Bibliothek, die ihn gerade implementiert.

| Verzeichnis | Inhalt |
|---|---|
| `kernels/quant/simulate/` | Triton-Kernel für simulierte Quantisierung mit Straight-Through-Backward auf jedem Gerät. Werden sowohl von QAT als auch von simulierter Post-Training-Quantisierung verwendet |
| `kernels/quant/execute/` | Pfade mit tatsächlicher Genauigkeit ausschließlich für finalisierte Modelle, ohne Backward: FP8-Tensor-Core-GEMM, dessen fusionierter Triton-Prolog und -Epilog sowie Kernel zum Entpacken gepackter Gewichte |
| `kernels/attention/` | Familienübergreifende Attention-Operationen: Slot `ms_deform_attn` und Richtlinie für fusionierte SDPA |

Die Grenze zwischen `simulate` und `execute` richtet sich danach, ob das Modell finalisiert ist, nicht danach, ob es trainiert oder bereitgestellt wird. Die Referenzimplementierungen verbleiben in `libreyolo/quant/` und definieren die numerische Bedeutung. `kernels/` beschleunigt sie lediglich. Für das Packen von Gewichten gibt es keine Varianten, da es Bestandteil des Checkpoint-Vertrags ist.

GEMM- und Attention-Slots besitzen keine Referenzimplementierung. Ein Aufrufer muss prüfen, ob `resolve()` einen Wert zurückgegeben hat, und einen eigenen portablen Pfad bereithalten. Deshalb enthalten ONNX-, TensorRT- und `torch.export`-Graphen immer die portable Mathematik.

### Auswahlüberschreibungen

`LIBREYOLO_KERNELS=off` oder `=reference` erzwingt Referenzimplementierungen und verhindert vollständig den Import beschleunigter Provider. Jeder andere Wert beschränkt die Auswahl auf Implementierungen, die unter diesem Namen registriert wurden. `LIBREYOLO_QUANT_KERNELS` wird als veralteter Alias aus der Zeit berücksichtigt, als sich die Registry unter `libreyolo/quant/` befand. Er wird nur gelesen, wenn `LIBREYOLO_KERNELS` nicht gesetzt ist. Beide stehen mit den übrigen Variablen unter [Einstellungen](/docs/reference/settings).

## Hub-Kernel

Auf dem Hugging Face Hub veröffentlichte kompilierte CUDA-Kernel werden zur Laufzeit über das optionale Paket `kernels` geladen. LibreYOLO bindet nichts davon direkt ein. Das Paket lädt das Artefakt und speichert es zwischen. Jeder Provider schreibt eine geprüfte Commit-Revision fest. Eine Aktualisierung dieses Pins erfordert vor der Aufnahme einen GPU-Paritätslauf.

Die Installation des Extras aktiviert die Funktion:

```bash
pip install "libreyolo[hub-kernels]"
```

Ohne das Paket ändert sich nichts und es erfolgt keine Netzwerkanfrage. `LIBREYOLO_HUB_KERNELS=0` deaktiviert das Abrufen ohne Deinstallation. Kann ein Kernel nicht geladen oder ausgeführt werden, deaktiviert er sich für den Rest des Prozesses und fällt mit einer einmaligen Warnung zurück.

Derzeit wird ein Slot vom Hub bereitgestellt: `ms_deform_attn`, der kompilierte Vorwärts- und Rückwärtsdurchlauf für mehrskalige deformierbare Attention aus Deformable DETR unter Apache 2.0. Er ist in der gesamten deformierbaren Abstammungslinie eingebunden: RF-DETR, Deformable DETR, DINO-DETR, LW-DETR, Grounding DINO, RT-DETR, RT-DETRv2, D-FINE, RT-DETRv4, DEIM, DEIMv2, EC und OV-DEIM. Da auch der Rückwärtsdurchlauf kompiliert ist, profitieren Training und Vorhersage.

Die Eignung ist bewusst eng gefasst. Eingaben müssen CUDA und float32 verwenden, die Ausführung muss im Eager-Modus erfolgen. Der Provider lehnt bei `torch.jit.is_tracing()`, `torch.compiler.is_compiling()`, `torch.compiler.is_exporting()` und `torch.onnx.is_in_onnx_export()` ab. Zwei Eingabelayouts fallen ebenfalls auf den portablen Pfad zurück: eine je Ebene unterschiedliche Punktanzahl sowie Sampling mit diskreten Ganzzahlindizes. Die Pose-Variante von EC ist nicht verbunden.

### Neu erreichbarer Kernel

Lies diesen Abschnitt, bevor du das Extra in einem bestehenden Projekt installierst.

In v1.4.0 wurde der Slot innerhalb einer Hilfsfunktion hinter einer Bedingung abgefragt, die fehlende Paare räumlicher Formen voraussetzte. RF-DETR reicht diese Paare immer durch seinen Decoder. Die Bedingung war deshalb nie erfüllt und der Kernel wurde in keinem Eager-Vorwärtsdurchlauf ausgeführt. In v1.5.0 wurde die Abfrage verschoben und der Kernel wird nun tatsächlich verwendet.

Die praktische Folge: Wenn du auf v1.5.0 aktualisierst und `libreyolo[hub-kernels]` unter CUDA installierst, verwenden RF-DETR und seine Abstammungslinie erstmals ein kompiliertes Binärprogramm für ihren Vorwärtsdurchlauf. Vorhersagen und Metriken können sich dadurch innerhalb der Gleitkommatoleranz verschieben. Eine Standardinstallation ohne das Extra ist nicht betroffen. Halte beim Vergleich von Metriken vor und nach dem Upgrade das Extra konstant oder setze auf beiden Seiten `LIBREYOLO_HUB_KERNELS=0`.

## Fusionierte Attention

Fusionierte Scaled Dot-Product Attention benötigt keine optionale Abhängigkeit, sondern nur unverändertes PyTorch. Ihre Verwendung wird deshalb durch eine Richtlinie statt durch Verfügbarkeit bestimmt. Es gelten zwei Regeln.

Erstens verwendet eine Graphaufzeichnung sie nie. Jede ausgetauschte Aufrufstelle behält hinter einer Exportprüfung die Gleichung aus primitiven Operationen. Dies deckt den ONNX-Export ab, dessen Standard-Opset kein SDPA-Symbol besitzt, sowie `torch.jit.trace`, das TorchScript, CoreML und NCNN verwenden. Dynamo-Aufzeichnungen liegen bewusst außerhalb dieser Sperre, da `torch.compile` SDPA besser als die manuelle Mathematik absenkt und Core AI sowie ExecuTorch SDPA selbst in Core ATen zerlegen.

Zweitens erfordert die Standardaktivierung bytegenaue Parität. Familien, die diese Hürde erfüllen, verwenden SDPA standardmäßig: SegFormer, Depth Anything und MoGe-2, BERT, Grounding DINO, SwinIR und PP-OCR. Familien ohne bytegenaue Parität behalten die manuelle Mathematik und stellen stattdessen ein Flag `fused_attn` bereit. Dieses schaltet `set_fused_attention(model)` um: Swin, der Swin-Backbone von DINO-DETR, BiRefNet und FeyNobg, OWLv2, LW-DETR, SigLIP 2, ZipDepth und MobileSAM. ViT und DeiT besitzen dasselbe Flag, aktivieren es entsprechend dem Upstream-Projekt aber standardmäßig. Derselbe Aufruf mit `enabled=False` deaktiviert es.

Die Aktivierung lohnt sich, wenn sie unterstützt wird. Auf einer RTX 5070 Ti unter FP16-Autocast sinkt die Swin-Window-Attention von 1,278 ms auf 0,721 ms, was einer Beschleunigung um den Faktor 1,77 entspricht. Die Vision-Attention von OWLv2 sinkt von 6,483 ms auf 1,735 ms, also um den Faktor 3,74.

## Hardware

| Plattform | Verhalten |
|---|---|
| CPU und MPS | Alle CUDA- und Triton-Prädikate schlagen fehl, sodass alles die Referenz verwendet |
| NVIDIA CUDA | Triton-Kernel sowie geeignete Hub- und GEMM-Kernel werden verwendet |
| AMD ROCm | Triton kann verwendet werden, da ROCm-Wheels das AMD-Backend von Triton enthalten. Die Parität wird in CI jedoch nur auf NVIDIA geprüft |

## Hinzufügen einer Implementierung

Rufe `register()` mit einem Namen und einem Prädikat auf. Externe kompilierte Kernel können als separates Paket `libreyolo_kernels` ausgeliefert werden, das sich beim Import selbst registriert. Dadurch bleibt ein privates Backend vollständig außerhalb des LibreYOLO-Baums.

Parität ist die Voraussetzung für jede integrierte Implementierung: exakte Übereinstimmung des Vorwärtsdurchlaufs mit der Referenz sowie Gradienten innerhalb von 1e-6 gegenüber dem Straight-Through-Estimator für die in der Testsuite enthaltenen Formen.

Die Kernelauswahl interagiert mit [CUDA-Graphen](/docs/reference/cuda-graphs). Die Inferenz-Paritätsmatrix wurde ohne installiertes Paket `kernels` ausgeführt. Die Aufzeichnungssicherheit bei aktivem kompiliertem Kernel wird davon nicht abgedeckt.

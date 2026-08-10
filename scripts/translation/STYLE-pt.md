# Portuguese translation style guide — LibreYOLO docs

How to produce `<slug>.pt.md` twins of the English docs in `content/docs/`.
The docs loader (`src/lib/docs.js`) picks up the twin automatically; the file
must be a complete, standalone copy of the English file with the rules below.

The `pt` locale is **Brazilian Portuguese (pt-BR)**. If a European Portuguese
locale is ever added it forks as `pt-PT` with its own file — so every place
where the two variants diverge is flagged in the glossary below, in the
`pt-PT` column. Never hedge between the two: write the Brazilian form and let
the flag carry the difference.

## Voice and register

- Brazilian Portuguese, post-Acordo Ortográfico spelling throughout. Address
  the reader as `você` (the standard register of modern Brazilian developer
  docs), or use impersonal constructions where they read better. Never `tu`,
  never `vós`, never mesoclisis (`far-se-á`).
- Brazilian clitic placement: proclisis by default (`se você quiser`,
  `isso se resolve em uma linha`), never the European enclitic default
  (`resolve-se`). Progressive is `está rodando`, not `está a rodar`.
- Post-Acordo means: no trema (`sequência`, not `seqüência`), no accent on
  `ideia`/`assembleia`, `voo`/`enjoo` unaccented, and Brazilian consonant
  clusters kept where Brazil kept them (`detecção`, `objecto` → `objeto`).
  Brazilian spelling keeps the `c` in `detecção`/`seleção`-type words exactly
  where the standard says so; pt-PT drops several of them (`deteção`).
- Technical, direct, unadorned — mirror the tone of the English source. Do not
  add, remove, summarize or "improve" content. Every paragraph, list item,
  table row, admonition and FAQ entry in the English file appears in the
  Portuguese file, in the same order.
- Spell out what the English spells out; keep the same level of formality.
  English docs here are terse and slightly informal; Portuguese should be too.
  Do not inflate into corporate register (`realizar a execução do modelo` for
  "run the model").

## What to translate

- Frontmatter: `title`, `seo_title`, `description`, `lead`, and `faq` (both
  `q` and `a`). Quote YAML strings that contain `:` or other special chars.
- Reader-visible prose values in `meta` and `verification` frontmatter blocks
  (labels like "Install" → "Instalação", prose values like "3.10 or newer" →
  "3.10 ou superior"). Values marked `mono` or that are code/identifiers stay.
- Snippet and tab `label` values that are descriptive phrases ("Video and
  streams" → "Vídeo e streams", "Check a dataset" → "Conferir um dataset").
  Labels that are proper names stay untouched: `Python`, `CLI`, `Bash`,
  library, tool and format names.
- `keywords`: **localize, don't translate literally.** Write the queries a
  Brazilian developer would actually type — a mix of Portuguese phrases and
  the English terms they'd realistically use (e.g.
  `"detecção de objetos python"`, `"treinar yolo dataset próprio"`,
  `"exportar yolo onnx"`). Keep model names as-is.
- Body prose, headings, table cells, image alt text, admonitions, and
  `hero.caption` (the media `src`/`poster` paths stay).
- Comments **inside** code blocks and snippets (`# download automático...`).

## What stays exactly as in English

- All code (statements, identifiers, strings, CLI commands, file names, URLs)
  — only comments change. Never translate string literals passed to code.
- Frontmatter keys themselves, their order, and the structural fields:
  `families`, `last_verified`, `snippets` structure (keys, order, count),
  `language` values, hero/media paths.
- Internal and external link **targets** (translate link text, never hrefs).
- Model names (YOLOv9, RF-DETR, RTMDet...), product names, library names,
  format names (ONNX, TensorRT, CoreML), and heading anchors implied by them.
- Numbers and units as written: `0.001 mAP`, `lr0=0.004`, `640`, `300 epochs`
  → `300 épocas`, but the numeral itself never changes and the decimal point
  stays a point (it is a metric value, not running prose).
- Acronyms and metric names: `mAP`, `mAP50`, `mAP50-95`, `IoU`, `NMS`, `FPS`,
  `ONNX`, `GPU`, `CPU`, `CUDA`, `AMP`, `FP16`, `INT8`, `SaaS`, `API`, `MIT`,
  `AGPL`.
- **English terms Brazilian devs keep untranslated.** Write these in English,
  unitalicized, with a Portuguese article: `o dataset`, `o checkpoint`, `o
  embedding`, `o backbone`, `o batch`, `o benchmark`, `o framework`, `o
  pipeline`, `o runtime`, `o deploy`, `o commit`, `o release`, `o stream`, `o
  bounding box`, `o ground truth`, `o learning rate`, `o overfitting`, `o
  early stopping`, `o warmup`, `o fine-tuning`, `o data augmentation`, `o
  matting`, `o clustering`, `o download`, `a flag`, `a label`, `a loss`.
  Translating these reads as machine translation to the target reader.
- Terms that **do** get translated even though the English is common in
  speech: `weights` → `pesos`, `layer` → `camada`, `threshold` → `limiar`,
  `mask` → `máscara`, `class` → `classe`, `epoch` → `época`,
  `training` → `treinamento`. Do not leave these in English.

## Glossary (use these consistently)

The third column flags where a future pt-PT fork must diverge. Blank means
both variants agree.

| English | Portuguese (pt-BR) | pt-PT would differ |
|---|---|---|
| library | a biblioteca (never "livraria" — false friend) | |
| framework | o framework | |
| file | o arquivo | **ficheiro** |
| weights | os pesos | |
| weights file | o arquivo de pesos | o ficheiro de pesos |
| checkpoint | o checkpoint | |
| dataset | o dataset | |
| training / to train | o treinamento / treinar | **o treino** |
| fine-tuning / to fine-tune | o fine-tuning / fazer fine-tuning (or ajuste fino) | |
| inference | a inferência | |
| computer vision | visão computacional | **visão por computador** |
| machine learning | machine learning (or aprendizado de máquina) | aprendizagem automática |
| object detection | detecção de objetos | **deteção de objetos** |
| bounding box | o bounding box (or a caixa delimitadora) | |
| oriented bounding box | a caixa orientada | |
| instance segmentation | segmentação de instâncias | |
| semantic segmentation | segmentação semântica | |
| panoptic segmentation | segmentação panóptica | |
| pose estimation | estimativa de pose | **estimação de pose** |
| depth estimation | estimativa de profundidade | estimação de profundidade |
| keypoints | os keypoints (pontos-chave) | |
| embedding | o embedding | |
| backbone | o backbone | |
| head (of a model) | a cabeça (do modelo) | |
| layer | a camada | |
| feature map | o mapa de características (feature map) | |
| threshold | o limiar (threshold) | |
| confidence | a confiança | |
| precision | a precisão | |
| recall | o recall | a revocação |
| accuracy | a acurácia | **a exatidão** |
| batch / batch size | o batch / o tamanho de batch | |
| epoch | a época | |
| learning rate | o learning rate (a taxa de aprendizado) | a taxa de aprendizagem |
| data augmentation | o data augmentation (o aumento de dados) | |
| mixed precision | a precisão mista | |
| early stopping | o early stopping | |
| overfitting | o overfitting | o sobreajuste |
| loss | a loss (a função de perda on first mention) | |
| quantization | a quantização | |
| export / to export | a exportação / exportar | |
| deployment / to deploy | o deploy / fazer deploy | a implementação / implementar |
| pretrained | pré-treinado | |
| label (annotation) | o rótulo (a label) | a etiqueta |
| annotation | a anotação | |
| ground truth | o ground truth | |
| class | a classe | |
| mask | a máscara | |
| open-vocabulary detection | detecção de vocabulário aberto | deteção de vocabulário aberto |
| anchor-free | sem âncoras (anchor-free) | |
| postprocessing / preprocessing | o pós-processamento / o pré-processamento | |
| edge device | o dispositivo de borda (edge) | o dispositivo periférico / edge |
| frame rate / FPS | os FPS (a taxa de quadros) | a taxa de fotogramas |
| streaming / stream | o streaming / o stream | |
| tiled inference | a inferência por blocos (tiles) | |
| download / to download | o download / baixar | a transferência / descarregar |
| cache / cached | o cache / em cache | a cache |
| screen / display | a tela | **o ecrã** |
| heatmap | o mapa de calor | |
| denoising | a remoção de ruído | |
| upscaling | o aumento de escala (upscaling) | |
| clustering | o clustering (o agrupamento) | |
| retrieval | a recuperação (a busca) | a pesquisa |
| open source | código aberto (open source) | |
| release | a versão (o release) | |
| runtime | o runtime | |
| pipeline | o pipeline | |
| benchmark | o benchmark | |

`mAP`, `mAP50`, `IoU`, `NMS`, `ONNX`, `GPU`, `CPU`, `CUDA` stay untouched.

Two traps worth naming: `library` is `biblioteca`, never `livraria`; and
`precision` vs `accuracy` are distinct metrics — `precisão` is reserved for
*precision*, so *accuracy* must be `acurácia` and never `precisão`.

## Formatting invariants (validated by script)

- Same number of headings, at the same levels, in the same order.
- Same number of fenced code blocks; code identical except comment lines.
- Same set of link targets.
- Same frontmatter keys as the English file — no keys added or dropped.
- Do not add translator notes, disclaimers, or a "translated from" line.

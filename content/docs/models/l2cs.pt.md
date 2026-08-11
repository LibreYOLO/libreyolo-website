---
title: L2CS-Net
families:
  - l2cs
seo_title: 'L2CS-Net: estimativa de olhar no LibreYOLO'
description: >-
  Use o L2CS-Net no LibreYOLO para estimativa de olhar em dois estágios com
  pitch e yaw. Instale, faça predições e exporte; o checkpoint do Gaze360 é
  somente para pesquisa.
lead: >-
  O L2CS-Net é um estimador de olhar em dois estágios: um detector de rostos
  localiza os rostos, e um tronco ResNet com duas cabeças de classificação por
  bins de ângulo prediz pitch e yaw para cada rosto. O LibreYOLO o encapsula
  somente para inferência.
keywords:
  - L2CS-Net
  - estimativa de olhar
  - rastreamento ocular python
  - direção do olhar
  - gaze estimation
  - Gaze360
  - detecção de rostos
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Sem face_detector: cai no detector de rostos embutido do OpenCV
        # (Haar no OpenCV 4, YuNet no OpenCV 5), então isso roda sem nenhum
        # download além do próprio checkpoint do L2CS.
        model = LibreYOLO("LibreL2CSr50.pt")
        result = model(SAMPLE_IMAGE)

        print(result.gaze.pitch, result.gaze.yaw)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreL2CSr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Fonte de rostos
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreL2CSr50.pt")

        # Passe ao L2CS boxes de um detector que você já rodou.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])

        # Ou nomeie um detector de rostos embutido específico.
        result = model(SAMPLE_IMAGE, face_detector="yunet")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreL2CSr50.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreL2CSr50.pt format=onnx
    - label: Usar o arquivo exportado
      language: python
      code: |
        import numpy as np
        import onnxruntime as ort

        # O grafo exportado é só o tronco ResNet e as duas cabeças de bins
        # de ângulo: ele recebe um recorte de rosto 448x448 pré-processado e
        # devolve (yaw_logits, pitch_logits) crus, não ângulos decodificados.
        # O softmax, a esperança sobre os bins e a conversão para graus ficam
        # no Python; veja libreyolo.models.l2cs.utils.bin_logits_to_angles.
        session = ort.InferenceSession("LibreL2CSr50.onnx")
        name = session.get_inputs()[0].name
        yaw_logits, pitch_logits = session.run(
            None, {name: np.zeros((1, 3, 448, 448), dtype=np.float32)}
        )
source_hash: 4ec43f4673b4be3e
---

## Instalação

O L2CS-Net não precisa de nenhum extra para construir, fazer predições ou
exportar um modelo do qual você já tenha um checkpoint.

```bash
pip install libreyolo
```

O único checkpoint que o LibreYOLO consegue buscar automaticamente, um
ResNet-50 treinado no Gaze360, é baixado via `gdown` em vez de um mirror HTTP
simples, porque ele fica no Google Drive do autor e não na org do LibreYOLO.
Esse caminho precisa do extra `gaze`:

```bash
pip install "libreyolo[gaze]"
```

Sem ele, o LibreYOLO imprime instruções de download manual em vez de falhar
silenciosamente.

## Predição

<code-tabs name="predict" />

O L2CS-Net é um estimador de dois estágios: um detector de rostos roda
primeiro, e a cabeça de olhar lê pitch e yaw de cada recorte de rosto que ele
devolve. Sem configuração, a predição recorre ao detector embutido do OpenCV,
então uma chamada simples funciona sem nenhum download adicional assim que você
tiver em mãos o próprio checkpoint do L2CS. `face_boxes` aceita boxes de um
detector que você já rodou; `face_detector` aceita `"auto"`, `"haar"`,
`"yunet"`, um modelo de detecção do LibreYOLO ou um callable simples.
`result.gaze` carrega pitch e yaw em radianos, alinhados linha a linha com
`result.boxes`, as caixas de rosto detectadas. Veja
[predição](/docs/predict) para fontes, streaming e tratamento de resultados.

## Variantes

Cinco profundidades de backbone compartilham uma mesma resolução de entrada e
aceitam os mesmos argumentos. O Gaze360, o dataset por trás do único checkpoint
publicado, treinou um ResNet-50; as outras quatro profundidades são suportadas
arquiteturalmente, mas não têm pesos publicados para carregar.

## Exportação

<export-matrix />

<code-tabs name="export" />

## Licenciamento

<provenance-box>

O LibreYOLO não hospeda nem espelha nenhum checkpoint do L2CS: não existe nada
dessa família na org do LibreYOLO no Hugging Face, diferente da maioria das
outras famílias deste site. O único checkpoint que a biblioteca consegue buscar
automaticamente vem direto da distribuição no Google Drive do próprio autor,
atrás do aviso de licença do Gaze360 impresso antes de a transferência começar,
e não é a cópia "republicada em huggingface.co/LibreYOLO" que o resumo acima dá
a entender.

</provenance-box>

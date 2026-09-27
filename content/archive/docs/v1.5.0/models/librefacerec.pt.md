---
title: LibreFaceRec
families:
  - facerec
seo_title: 'LibreFaceRec: reconhecimento e verificação facial'
description: >-
  Use o LibreFaceRec no LibreYOLO para detecção de rostos, embedding e
  verificação facial. Instale e faça a predição; os pesos de embedding são
  Apache-2.0.
lead: >-
  O LibreFaceRec é a tarefa de embedding facial do LibreYOLO: um detector de
  rostos localiza e alinha os rostos, e uma cabeça de reconhecimento produz um
  embedding de identidade normalizado por L2 para verificação ou busca.
keywords:
  - LibreFaceRec
  - reconhecimento facial
  - reconhecimento facial python
  - embedding facial
  - verificação facial
  - comparar dois rostos
  - ArcFace
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # nomes librefacerec-* apontam para esta família independentemente do
        # sufixo do arquivo e são baixados da org do LibreYOLO no Hugging Face
        # no primeiro uso, junto com o detector de rostos padrão.
        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)   # (N, D), normalizado por L2
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=face.jpg
    - label: Verificar
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        # Compara o rosto mais proeminente de cada imagem pela similaridade
        # de cosseno dos seus embeddings normalizados por L2.
        result = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)
        print(result["similarity"], result["same_person"])
    - label: Busca na galeria
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        query = model("query.jpg").embeddings          # os rostos desta imagem
        gallery = model.embed(["a.jpg", "b.jpg", "c.jpg"])   # (N_total, D)

        # similaridades de cosseno (query_faces, N_total).
        scores = query.similarity(gallery)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")
        model.export(format="onnx")
source_hash: f1a345bb96e32f12
---

## Instalação

A cabeça de reconhecimento do LibreFaceRec roda através do `onnxruntime`, que
não faz parte da instalação base.

```bash
pip install "libreyolo[onnx]"
```

## Predição

<code-tabs name="predict" />

Detecção e reconhecimento são dois grafos ONNX separados por trás de uma única
chamada: um detector de rostos localiza e alinha cada rosto em um recorte
canônico, e a cabeça de reconhecimento devolve um embedding normalizado por L2
para cada rosto. Sem nenhuma configuração, o `predict()` baixa e pareia
automaticamente o detector padrão que vem embutido. `face_detector` aceita um
callable, um modelo de detecção do LibreYOLO ou uma instância de
`FaceDetector`; `face_boxes` pula a detecção por completo usando caixas que
você já tem. `result.embeddings` guarda uma linha por rosto detectado, alinhada
com `result.boxes`; o método `.similarity()` dele calcula a similaridade de
cosseno contra outro embedding ou contra uma galeria inteira em uma única
chamada. Para comparar duas imagens diretamente, em vez de dois embeddings já
calculados, `model.verify(image_a, image_b)` roda detecção e embedding nas duas
e compara o rosto mais confiante de cada uma. Qualquer outro modelo ONNX de
reconhecimento na convenção ArcFace (recorte alinhado na entrada, embeddings
`(N, D)` na saída) pode ser usado no lugar, passando o caminho do arquivo dele
em vez de um nome `librefacerec-*`. Veja [predição](/docs/predict) para fontes,
streaming e tratamento de resultados.

## Exportação

<export-matrix />

O LibreFaceRec já embrulha um grafo ONNX pré-exportado; reexportá-lo para outro
formato não está implementado.

## Licenciamento

<provenance-box>

O detector de rostos padrão que vem embutido é um segundo artefato sob uma
segunda licença: o YuNet do OpenCV Zoo, MIT, copyright Shiqi Yu. Nenhum código
de arquitetura é portado de qualquer um dos dois projetos; ambos os grafos são
consumidos de forma opaca através do `onnxruntime`, então o wrapper do próprio
LibreYOLO não carrega nenhum código de terceiros e é MIT por inteiro.

</provenance-box>

## Citação

<citation-block />

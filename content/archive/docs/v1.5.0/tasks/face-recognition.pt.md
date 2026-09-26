---
title: Reconhecimento facial
seo_title: Reconhecimento facial no LibreYOLO
description: >-
  Detecte, gere embeddings e identifique rostos no LibreYOLO. Cadastre uma
  galeria, compare duas imagens e faça a correspondência por similaridade de
  cosseno, em Python ou pela CLI.
lead: >-
  O reconhecimento facial é a tarefa embed aplicada a rostos. Um detector
  localiza e alinha cada rosto, uma cabeça de reconhecimento devolve um vetor
  normalizado por L2 para cada rosto, e a identidade é decidida por similaridade
  de cosseno em relação a referências cadastradas, e não por uma lista fixa de
  classes.
keywords:
  - reconhecimento facial python
  - embedding facial
  - verificação facial python
  - comparar dois rostos
  - identificar pessoas em fotos
  - arcface onnx
  - similaridade de cosseno rostos
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Os nomes librefacerec-* apontam para a família de embeddings faciais

        # seja qual for o sufixo do arquivo, e são baixados da org da LibreYOLO

        # no Hugging Face no primeiro uso, junto com o detector de rostos
        padrão.

        model = LibreYOLO("librefacerec-l.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)             # (N, 4) boxes de rostos

        print(result.embeddings.data.shape)  # (N, D), uma linha por rosto

        print(result.embeddings.dim)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=photo.jpg
    - label: Comparar duas imagens
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("librefacerec-l.onnx")


        # Roda detecção e embedding nas duas imagens e compara o rosto

        # com maior confiança de cada uma. A similaridade de cosseno está em
        [-1, 1].

        outcome = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)

        print(outcome["similarity"], outcome["same_person"])
    - label: Cadastrar uma galeria e identificar
      language: python
      code: |
        from libreyolo import Gallery, LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        gallery = Gallery(model)
        gallery.enroll("ada", ["people/ada/1.jpg", "people/ada/2.jpg"])
        gallery.enroll("grace", "people/grace/1.jpg")
        gallery.save("faces.npz")

        result = model("group_photo.jpg", gallery=gallery, threshold=0.4)
        for name, score in result.identities.data:
            print(name, score)   # name é None abaixo do limiar
    - label: Cadastrar e identificar pela CLI
      language: bash
      code: >
        libreyolo enroll model=librefacerec-l.onnx source=people/
        gallery=faces.npz

        libreyolo predict model=librefacerec-l.onnx source=group_photo.jpg
        gallery=faces.npz
    - label: Usar seus próprios boxes de rostos
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("librefacerec-l.onnx")


        # face_boxes pula a detecção por completo; face_detector aceita um

        # callable, um modelo de detecção LibreYOLO ou uma instância de
        FaceDetector.

        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])

        print(result.embeddings.data.shape)
source_hash: d7dfcb6f812ebb2d
---

## Definição

O reconhecimento facial devolve um vetor por rosto, não uma label. A predição
roda em duas etapas: um detector de rostos localiza cada rosto e seus cinco
landmarks, o recorte é transformado para um alinhamento canônico de 112x112, e
uma cabeça de reconhecimento emite um embedding normalizado por L2.

`result.embeddings` é um payload `Embeddings` de shape `(N, D)`, alinhado linha a
linha com `result.boxes`, então a linha `i` descreve o rosto do box `i`. Como as
linhas são vetores unitários, a similaridade de cosseno é um produto escalar, e
`embeddings.similarity()` a calcula em relação a outro `Embeddings` ou a uma
matriz inteira em uma única chamada.

Dar nome a um rosto é uma etapa separada. Uma `Gallery` guarda vetores de
referência nomeados; passar `gallery=` para `predict()` anexa
`result.identities`, alinhado linha a linha com os embeddings, com um nome e o
melhor score de cosseno por rosto. Um rosto abaixo do limiar de correspondência
mantém `None` como nome, e o nome mais próximo abaixo do limiar nunca é colocado
no lugar.

A chave de tarefa canônica da biblioteca é `embed`. `face-recognition`,
`facial-recognition`, `reid` e `face` são todas normalizadas para ela, então
`task="face-recognition"` e `task="embed"` selecionam a mesma coisa. Os rostos
são o formato por região dessa tarefa mais ampla; [embeddings](/docs/tasks/embeddings)
cobre os formatos de imagem inteira e de texto, a API compartilhada de
`Embeddings`, `Identities` e `Gallery`, e os modelos que produzem vetores sem
detectar nada.

## Modelos

[LibreFaceRec](/docs/models/librefacerec) é a família desta tarefa. São dois
artefatos ONNX por trás de uma única chamada: `librefacerec-l.onnx`, uma cabeça
de reconhecimento iResNet100 que produz embeddings de 512-d, e
`librefacerec-det.onnx`, o detector de rostos padrão com cinco landmarks, vindo
do OpenCV zoo. Ambos são baixados da org da LibreYOLO no Hugging Face no primeiro
uso. Qualquer outro arquivo ONNX na convenção ArcFace (entrada alinhada de
112x112, saída `(N, D)`) pode substituir a cabeça de reconhecimento se você
passar o caminho dele em vez de um nome `librefacerec-*`.

A chave de tarefa `embed` é mais ampla que rostos. [CLIP](/docs/models/clip),
[SigLIP2](/docs/models/siglip2) e [DINOv2](/docs/models/dinov2) também suportam
`task="embed"` e devolvem um vetor por imagem inteira, o que é recuperação de
imagens e não identidade facial. Eles compartilham a API de `Gallery` e
`Embeddings`, então o fluxo de cadastro e correspondência descrito abaixo também
vale, mas eles não detectam nem alinham rostos.

A cabeça de reconhecimento roda em cima do `onnxruntime`, que a instalação base
não traz:

```bash
pip install "libreyolo[onnx]"
```

## Predição

<code-tabs name="predict" />

Se você não mexer em nada, `predict()` baixa e emparelha o detector padrão.
`face_detector` o substitui por um callable, um modelo de detecção LibreYOLO ou
uma instância de `FaceDetector`, e pode ser definido no construtor ou por
chamada. `face_boxes` contorna a detecção com boxes que você já tem. Na CLI,
`face_detector=` aceita o caminho de um `.onnx` de detector de rostos ou o nome
de um detector LibreYOLO.

`model.verify(image_a, image_b)` é o atalho para duas imagens: ele gera o
embedding do rosto com maior confiança de cada uma e devolve
`{"similarity", "same_person", "threshold"}`. `model.embed(sources)` devolve
todas as linhas de rostos de uma ou mais imagens empilhadas em um único tensor
`(N_total, D)`. Veja [predição](/docs/predict) para fontes, streaming e
tratamento de resultados.

## Formato do dataset

O cadastro lê uma pasta por identidade. O nome da pasta vira a identidade, e
cada imagem dentro dela contribui com referências para esse nome:

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

`libreyolo enroll` percorre essa árvore e escreve uma galeria `.npz`. Um arquivo
de galeria já existente é estendido no lugar em vez de substituído, então dá
para adicionar identidades ao longo do tempo. As galerias ficam vinculadas aos
pesos que as produziram pela dimensão do embedding e por uma impressão digital
do arquivo; fazer a correspondência com um modelo diferente levanta um erro em
vez de comparar espaços vetoriais incompatíveis.

Por padrão cada imagem de origem contribui com uma linha de referência, a do
rosto com maior confiança, então um retrato com pessoas ao fundo cadastra apenas
o retratado. Passe `select="all"` para `Gallery.enroll` para armazenar todas as
linhas devolvidas.

## Treinamento

Nenhuma família desta tarefa treina dentro do LibreYOLO.
`LibreFaceEmbedder.train()` levanta um erro: treine uma cabeça de reconhecimento
por fora, exporte-a para ONNX na convenção ArcFace e carregue o arquivo pelo
caminho.

## Validação

Não há validador de dataset para esta tarefa, e `val()` levanta um erro em vez de
fingir o contrário. A acurácia de verificação é medida com `model.verify()` em
pares de imagens rotulados, varrendo `threshold` para escolher o ponto de
operação que você quer. A acurácia de identificação é medida cadastrando uma
galeria e lendo `result.identities.name` e `result.identities.score` em imagens
separadas, contando um nome `None` como rejeição.

## Exportação

A cabeça de reconhecimento já é um grafo ONNX, então não há nada a converter:
`LibreFaceEmbedder.export()` levanta um erro. Faça deploy do arquivo `.onnx`
diretamente, ou aponte o LibreYOLO para ele e deixe a família cuidar da detecção,
do alinhamento e da normalização.

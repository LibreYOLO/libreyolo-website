---
title: Estimativa do olhar
seo_title: Estimativa do olhar no LibreYOLO
description: >-
  Estime o pitch e o yaw do olhar de cada rosto no LibreYOLO. Faça predições
  pelo Python ou pela CLI, leia os ângulos em radianos e exporte a cabeça de
  olhar para ONNX.
lead: >-
  A estimativa do olhar retorna uma direção de olhar para cada rosto de uma
  imagem. O LibreYOLO modela isso como uma tarefa de duas etapas: um detector de
  rostos roda primeiro, e uma cabeça de olhar lê o pitch e o yaw de cada recorte
  de rosto que ele retorna.
keywords:
  - estimativa de olhar python
  - eye tracking python
  - rastreamento ocular
  - direção do olhar
  - pitch yaw olhar
  - L2CS-Net
  - pose da cabeça
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Sem face_detector, a predição recorre ao detector embutido do
        # OpenCV, então nada é baixado além do checkpoint.
        model = LibreYOLO("LibreL2CSr50.pt")
        result = model(SAMPLE_IMAGE)

        gaze = result.gaze
        print(gaze.pitch, gaze.yaw)              # radianos, uma linha por rosto
        print(gaze.pitch_deg, gaze.yaw_deg)      # os mesmos ângulos em graus
        print(gaze.direction_3d)                 # vetores unitários (N, 3)
    - label: CLI
      language: bash
      code: >
        # Ao contrário do caminho em Python, a CLI não tem fallback automático:

        # modelos de olhar exigem um detector de rostos explícito, e ele precisa

        # ser um detector do LibreYOLO cujos boxes sejam rostos.

        libreyolo predict model=LibreL2CSr50.pt source=photo.jpg
        face_detector=face-detector.pt save=True
    - label: Escolher a fonte dos rostos
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreL2CSr50.pt")


        # Passe para a cabeça de olhar os boxes de um detector que você já
        rodou.

        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])


        # Ou nomeie um dos detectores embutidos.

        result = model(SAMPLE_IMAGE, face_detector="yunet")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreL2CSr50.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreL2CSr50.pt format=onnx
source_hash: 22aa3c3d87b0c730
---

## Definição

A estimativa do olhar retorna dois ângulos por rosto. `result.gaze` é um payload
`Gaze` de formato `(N, 2)`, com o pitch na coluna 0 e o yaw na coluna 1, em
radianos, alinhado linha a linha com `result.boxes`, os boxes de rosto
detectados. A convenção é a que o L2CS-Net usa: um yaw positivo gira o olhar
para a esquerda do sujeito, e um pitch positivo gira o olhar para baixo.

O mesmo payload expõe `pitch_deg` e `yaw_deg` para graus, e `direction_3d`, um
vetor unitário `(N, 3)` no referencial da câmera com as colunas `(x, y, z)`.

Como a tarefa é de duas etapas, uma predição depende de dois modelos. Rostos que
o detector deixa passar não têm linha de olhar, e boxes que ele posiciona mal
produzem ângulos a partir de um rosto mal recortado. A chave canônica da tarefa é
`gaze`; `gaze-estimation` é normalizado para ela.

## Modelos

A [L2CS-Net](/docs/models/l2cs) é a única família que atende esta tarefa. Ela
combina um tronco ResNet com duas cabeças paralelas de classificação por bins de
ângulo, uma para o pitch e outra para o yaw, sobre recortes de rosto de 448x448.
A arquitetura dá suporte a cinco profundidades de backbone, e uma delas, a
ResNet-50, tem checkpoint publicado.

Os pesos carregam uma restrição de licença. Foram treinados no Gaze360, cuja
licença permite apenas uso em pesquisa e uso não comercial e proíbe a
redistribuição, então o LibreYOLO não espelha nada para essa família. O único
checkpoint que a biblioteca consegue baixar automaticamente vem direto da
distribuição dos próprios autores no Google Drive, via `gdown`, depois de
imprimir os termos da licença. Leia [L2CS-Net](/docs/models/l2cs) antes de fazer
deploy dele.

Esse caminho de download precisa do extra `gaze`:

```bash
pip install "libreyolo[gaze]"
```

Sem ele, a biblioteca imprime instruções de download manual em vez de tentar a
transferência. Fazer predição e exportação com um checkpoint que você já tem
não exige extra nenhum.

## Predição

<code-tabs name="predict" />

A fonte dos rostos é escolhida de uma destas três formas. `face_boxes` passa
boxes que você já calculou e pula a detecção. `face_detector` aceita `"auto"`,
`"haar"`, `"yunet"`, um modelo de detecção do LibreYOLO ou um callable comum, e
pode ser definido no construtor ou a cada chamada. Se você não definir nada em
Python, a predição recorre ao detector embutido do OpenCV, então uma chamada
simples funciona sem configurar nada. No OpenCV 4 esse detector é a cascata de
Haar que vem dentro do wheel e não precisa de download nenhum; no OpenCV 5, onde
a API do Haar foi removida, é o YuNet, que baixa uma única vez um arquivo de
modelo pequeno do zoo do OpenCV.

A CLI não compartilha esse fallback. `libreyolo predict` rejeita um modelo de
olhar sem `face_detector=`, e o valor que ele aceita é o nome de um detector do
LibreYOLO ou o caminho de um checkpoint. Veja [predição](/docs/predict) para
fontes, streaming e tratamento de resultados.

## Treinamento

Nenhuma família desta tarefa treina dentro do LibreYOLO. `LibreL2CS.train()`
lança uma exceção: treine no projeto original do L2CS-Net e carregue aqui o
state dict resultante.

## Validação

Validar contra datasets de ground truth de olhar está fora do escopo, e `val()`
lança uma exceção em vez de retornar métricas que não calculou. Não existe
dicionário `metrics/` para esta tarefa. Faça a avaliação no projeto original,
com o dataset para o qual o checkpoint foi treinado.

## Exportação

<code-tabs name="export" />

O contrato de exportação do olhar cobre ONNX, TorchScript, ExecuTorch, TensorRT
e OpenVINO. O que sai da biblioteca é apenas o tronco ResNet e as duas cabeças
de bins de ângulo: o grafo recebe um recorte de rosto 448x448 já pré-processado
e retorna os logits brutos de yaw e pitch. A detecção de rostos, o recorte, o
softmax, o valor esperado sobre os bins e a conversão para ângulos ficam todos em
Python, em `libreyolo.models.l2cs.utils`. Veja [exportação](/docs/export) para os
formatos e seus argumentos.

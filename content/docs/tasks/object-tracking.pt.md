---
title: Rastreamento de objetos
seo_title: Rastreamento de objetos no LibreYOLO
description: >-
  Rastreie objetos ao longo dos frames de um vídeo no LibreYOLO com ByteTrack,
  BoT-SORT, OC-SORT ou Deep OC-SORT, sobre qualquer modelo de detecção,
  segmentação ou pose.
lead: >-
  O rastreamento atribui uma identidade estável a cada detecção ao longo dos
  frames de um vídeo. O LibreYOLO não modela isso como uma tarefa com pesos
  próprios: é um modo de predição, model.track(), que roda o tracker escolhido
  sobre a saída por frame de um modelo de detecção, segmentação ou pose.
keywords:
  - rastreamento de objetos python
  - multi object tracking video
  - bytetrack
  - botsort
  - ocsort
  - deep ocsort
  - track id yolo
  - tracking com reid
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # track() é um gerador: um Results por frame processado.
        for result in model.track("video.mp4"):
            print(result.track_id)        # tensor int (N,), alinhado com boxes
            print(result.boxes.xyxy)
    - label: Escolher um tracker
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # "bytetrack" (padrão), "botsort", "ocsort" ou "deepocsort".
        for result in model.track("video.mp4", tracker="botsort"):
            print(result.track_id)
    - label: Salvar um vídeo anotado
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Sem output_path, o arquivo vai parar em runs/track/<video_stem>.mp4.
        for result in model.track("video.mp4", save=True, vid_stride=2):
            pass
    - label: Ajustar um tracker
      language: python
      code: >
        from libreyolo import BoTSortConfig, LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # O tipo da config seleciona o tracker, então tracker= é redundante
        aqui.

        config = BoTSortConfig(track_buffer=60, frame_rate=25, enable_cmc=False)

        for result in model.track("video.mp4", tracker_config=config):
            print(result.track_id)

        # Ou passe os mesmos campos como keyword arguments e deixe track()
        montá-la.

        for result in model.track("video.mp4", tracker="botsort",
        track_buffer=60):
            print(result.track_id)
source_hash: f1fa7dcf60597d6b
---

## Definição

O rastreamento não é uma das chaves de tarefa do LibreYOLO, e não há checkpoint
de rastreamento para baixar. É um método do modelo, `model.track(source)`, que
roda a detecção em cada frame e associa os resultados ao longo do tempo. O
método é um gerador: ele emite um `Results` por frame processado, com
`result.track_id` definido como um tensor de inteiros `(N,)` alinhado com
`result.boxes`. Os mesmos IDs também estão em `result.boxes.id`.

Só objetos confirmados e atualmente rastreados são emitidos. Um track que a
associação perde continua vivo por um número configurado de frames antes de ser
descartado, `track_buffer` para ByteTrack e BoT-SORT e `max_age` para as duas
variantes do OC-SORT, então um objeto recuperado dentro dessa janela mantém o
ID original.

Como a associação acontece depois da detecção, os demais payloads do frame
sobrevivem a ela: o `Results` rastreado é o `Results` da detecção fatiado nas
linhas correspondentes, então máscaras e keypoints vêm junto com os boxes.

## Modelos

Duas escolhas independentes entram em uma execução de rastreamento: o modelo que
produz os boxes a cada frame, e o tracker que os liga.

Qualquer modelo nativo do LibreYOLO cuja tarefa seja detecção, segmentação ou
pose expõe `track()`, então a escolha do detector é a de sempre. Veja
[o índice de modelos](/docs/models) para a lista completa, ou comece por
[YOLO9](/docs/models/yolov9), [RF-DETR](/docs/models/rf-detr),
[D-FINE](/docs/models/d-fine) ou [RTMDet](/docs/models/rtmdet). Tarefas cujos
resultados não têm box para associar recusam a chamada em vez de devolver IDs
sem sentido: classificação, caixas orientadas, pontos, profundidade, normais de
superfície, bordas, segmentação semântica e panóptica, restauração, OCR e malha
corporal, todas lançam exceção em `track()`.

Dois dos níveis de modelo do LibreYOLO também não o oferecem. Modelos
carregados por `LibreSAM` são segmentadores de imagem, e modelos carregados por
`LibreOpenVocab` são detectores por frame; ambos lançam exceção em `track()` e
são usados com `predict()` por frame no lugar disso.

O rastreamento roda em modelos PyTorch nativos. Um artefato exportado carregado
por `LibreYOLO("model.onnx")` devolve um objeto de backend de runtime, que traz
`predict()` mas não `track()`.

Quatro trackers acompanham a biblioteca, selecionados pelo argumento `tracker`:

`"bytetrack"` é o padrão. Usa apenas movimento, com um filtro de Kalman e uma
associação em três estágios: primeiro as detecções de alta confiança, depois uma
segunda passada que dá às detecções de baixa confiança a chance de parear com um
track existente antes de serem descartadas, e então os tracks não confirmados.
Configurado com `TrackConfig`.

`"botsort"` mantém o ciclo de vida em três estágios do ByteTrack, mas usa um
estado de Kalman de centro-largura-altura e compensa os tracks previstos pelo
movimento da câmera antes do pareamento. Esta é a variante do BoT-SORT que usa
apenas movimento; ela não roda nenhum modelo de aparência. Configurado com
`BoTSortConfig`, que acrescenta `enable_cmc`, `cmc_method` e `cmc_downscale`.

`"ocsort"` também usa apenas movimento, e acrescenta um termo de direção da
velocidade ao custo de associação, uma segunda passada de associação contra a
última observação real de cada track, e uma suavização do estado de Kalman ao
longo de uma trajetória virtual quando um track é reencontrado. Configurado com
`OCSortConfig`.

`"deepocsort"` estende o OC-SORT com aparência. Cada track guarda uma média
móvel ponderada pela confiança de embeddings de reidentificação, e um termo de
similaridade de cosseno se junta ao custo de associação, então as identidades
sobrevivem a oclusões longas e a alvos que se cruzam. Custa um forward pass de
uma pequena rede de embedding a cada frame, e os pesos OSNet dela são baixados
no primeiro uso. Configurado com `DeepOCSortConfig`.

## Predição

<code-tabs name="predict" />

`track_conf` define o limiar do primeiro estágio de associação:
`track_high_thresh` para ByteTrack e BoT-SORT, `det_thresh` para OC-SORT e Deep
OC-SORT. Não é o `conf` do `predict()`, e para ByteTrack, BoT-SORT e OC-SORT o
detector roda internamente com um limiar mais baixo para que detecções fracas
continuem disponíveis para a passada de recuperação. O Deep OC-SORT roda o
detector no próprio `det_thresh`. Para ByteTrack e BoT-SORT, `track_conf`
precisa ser igual ou superior a `track_low_thresh`, cujo padrão é 0.1.

As configurações do tracker chegam de duas formas. Passe uma instância de config
para `tracker_config=`, e o tipo dela seleciona o tracker, tornando `tracker=`
redundante. Ou passe os campos como keyword arguments e deixe `track()` montar a
config para o tracker que você nomeou; chaves desconhecidas emitem aviso em vez
de serem aplicadas silenciosamente. De qualquer forma, `track_conf` é ignorado
quando a chave correspondente é definida explicitamente.

Os demais argumentos espelham a predição: `iou`, `imgsz`, `classes`, `max_det`,
`vid_stride`, `show`, e `save` com `output_path`. A fonte é o caminho de um
arquivo de vídeo. Veja [predição](/docs/predict) para o tratamento dos
resultados.

## Treinamento

Trackers não são treinados. Três dos quatro são modelos puramente de movimento,
sem nenhum parâmetro aprendido, e a rede de aparência do Deep OC-SORT é um
checkpoint de reidentificação publicado que é baixado no primeiro uso. Melhorar
a qualidade do rastreamento significa melhorar o detector, ou ajustar os limiares
de associação acima.

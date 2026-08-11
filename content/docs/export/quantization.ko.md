---
title: 양자화
seo_title: PyTorch에서 LibreYOLO 모델 양자화
description: >-
  LibreYOLO의 PyTorch 양자화 API를 설명합니다. 9개 레시피, 학습 데이터와 분리된 보정, QAT 및 QAD, 배포 아티팩트
  2개를 다룹니다.
lead: >-
  LibreYOLO의 양자화는 전부 PyTorch에서 실행됩니다. model.quantize()는 모델의 Conv2d 및 Linear 모듈을
  양자화된 동등 모듈로 바꾸고 보정합니다. 결과는 일반적인 predict, val, train, save 계약을 유지하므로 양자화된 모델은
  부동소수점 모델과 같은 검증기로 평가됩니다.
keywords:
  - libreyolo 양자화
  - int8 ptq
  - 양자화 인식 학습
  - qat qad
  - nvfp4 mxfp4
  - fp8 e4m3
  - 보정 데이터셋
  - qdq onnx 변환
last_verified: 1.5.0
meta:
  - label: 호출
    value: 'model.quantize(recipe="int8", calib="coco128.yaml")'
    mono: true
  - label: 명령
    value: libreyolo quantize --model M.pt --recipe int8 --calib coco128.yaml
    mono: true
  - label: 추가 설치
    value: 없습니다. 양자화는 PyTorch에서 실행됩니다.
  - label: 계열
    value: 'yolo9, rfdetr, birefnet, feynobg'
  - label: 레시피
    value: 'fp16, bf16, fp8, int8, w4a16, w4a8, nvfp4, mxfp4, int2'
    mono: true
  - label: 배포 아티팩트
    value: '패킹된 체크포인트에는 export(format="pt"), QDQ INT8 그래프에는 export(format="onnx")'
    mono: true
verification: >-
  dev 브랜치의 libreyolo/quant/api.py, libreyolo/models/base/model.py,
  libreyolo/cli/commands/quantize.py, docs/quantization.md를 확인했습니다. 체크포인트 크기 수치는
  docs/quantization.md에 기록된 측정값입니다.
snippets:
  quantize:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # 구조 교체 및 보정입니다. calib은 활성화 범위와 스케일을 계산하기 위해

        # 순전파로만 읽는 소규모 비레이블 이미지 집합입니다.

        qmodel = model.quantize(recipe="int8", calib="coco128.yaml",
        samples=128)


        print(qmodel.quant_info())

        qmodel.val(data="coco8.yaml")          # 부동소수점 모델과 같은 검증기

        qmodel.save("LibreYOLO9s-int8.pt")     # 체크포인트에 양자화 매니페스트가 포함됩니다
    - label: CLI
      language: bash
      code: >
        libreyolo quantize --model LibreYOLO9s.pt --recipe int8 --calib
        coco128.yaml
    - label: 인수
      language: python
      code: |
        model.quantize(
            recipe="int8",
            calib="coco128.yaml",      # data.yaml 경로 또는 내장 이름이며 None은 보정을 건너뜁니다
            samples=128,               # 최대 보정 이미지 수
            batch=8,                   # 보정 배치 크기
            algorithm="auto",          # auto와 minmax는 같으며 percentile이 대안입니다
            keep_high_precision=None,  # None은 계열 정책을 사용합니다
            verbose=True,
        )
  reload:
    - label: 양자화된 체크포인트로 다시 불러오기
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 가중치를 불러오기 전에 양자화 매니페스트가 양자화된 구조와
        # 스케일을 다시 빌드합니다.
        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")
        print(qmodel.quant_info())
  train:
    - label: 양자화된 모델에서 일반 train()으로 QAT 실행
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        # 처음부터 실행하는 학습이 아니라 파인튜닝이므로 파인튜닝 학습률을 사용합니다.
        qmodel.train(data="coco8.yaml", epochs=5, lr0=1e-4)
    - label: 기존 증류 인수로 QAD 추가
      language: python
      code: |
        qmodel.train(
            data="coco8.yaml",
            epochs=5,
            lr0=1e-4,
            distill_model="LibreYOLO9m.pt",
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train --model LibreYOLO9s-int8.pt --data coco8.yaml --epochs 5
        --lr0 1e-4
  export:
    - label: 패킹된 PyTorch 체크포인트
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        # LibreYOLO9s-int8-final.pt를 작성합니다. 저비트 가중치와 스케일을 패킹하고
        # fp32 마스터를 제거하며 양자화되지 않은 나머지를 fp16으로 변환합니다.
        qmodel.export(format="pt")

        # remainder="fp32"는 양자화되지 않은 텐서를 정확히 유지합니다.
        qmodel.export(format="pt", remainder="fp32")
    - label: QDQ INT8 ONNX
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        # 모델 자체의 보정 또는 QAT 학습 스케일을 담은 그래프 내
        # QuantizeLinear/DequantizeLinear 쌍입니다.
        qmodel.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9s-int8.pt --format onnx
  dequantize:
    - label: QAT 학습 가중치를 유지하며 부동소수점으로 복원
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")
        qmodel.dequantize()

        # 이제 모든 부동소수점 내보내기를 지원하는 정밀도로 적용할 수 있습니다.
        qmodel.export(format="tensorrt", half=True)
source_hash: 4ffb06b87cad017e
---

## 설치

양자화에는 extra가 필요하지 않습니다. 모듈 교체, 보정 패스, 시뮬레이션된 산술은
모두 PyTorch에서 실행되므로 `pip install libreyolo`가 전체 요구 사항입니다. 배포
아티팩트에는 자체 형식의 요구 사항이 필요하며 ONNX 경로에서는
`libreyolo[onnx]`입니다.

## 양자화

<code-tabs name="quantize" />

`quantize()`는 불러온 모델을 제자리에서 변환하고 반환합니다. 그래디언트는 관여하지
않습니다. 교체 단계가 양자화 모듈을 설치하고 보정 패스는 순전파만 실행합니다.

생성된 체크포인트는 `quant` 매니페스트가 첨부된 일반 LibreYOLO 체크포인트이므로
구조와 스케일을 온전히 유지한 채 다시 불러옵니다.

<code-tabs name="reload" />

QAT 실행 중 작성된 트레이너 체크포인트에도 매니페스트가 있으므로 해당 실행의
`best.pt` 자체가 양자화된 체크포인트입니다.

## 레시피

지원하는 계열은 `yolo9`, `rfdetr`, `birefnet`, `feynobg` 4개입니다.

| 레시피 | 동작 | 계열 | 보정 |
|---|---|---|---|
| `fp16` | float32 입출력 계약과 함께 반정밀도로 변환합니다. 추론 전용입니다. | 4개 모두 | 없음 |
| `bf16` | float32의 지수 범위를 유지하는 bfloat16으로 변환합니다. DETR 계열 모델에서 fp16 오버플로가 발생할 때 해결책입니다. 추론 전용입니다. | 4개 모두 | 없음 |
| `fp8` | `Conv2d`와 `Linear`의 E4M3 가중치 및 활성화입니다. 채널별 가중치 스케일과 보정된 텐서별 활성화 스케일을 사용합니다. | 4개 모두 | 필수 |
| `int8` | `Conv2d`와 `Linear`의 W8A8입니다. 채널별 대칭 가중치와 텐서별 affine 활성화를 사용합니다. | 4개 모두 | 필수 또는 가중치 전용 `calib=None` |
| `w4a16` | `Linear`에서 `in_features` 방향 그룹 128의 그룹 대칭 INT4 가중치와 부동소수점 활성화를 사용합니다. | rfdetr, birefnet, feynobg | 불필요 |
| `w4a8` | `Linear`에서 그룹 INT4 가중치와 보정된 INT8 활성화를 사용합니다. | rfdetr, birefnet, feynobg | 필수 |
| `nvfp4` | `Linear`의 W4A4 NVFP4입니다. E2M1 요소, 16요소 블록, FP8 E4M3 블록 스케일, FP32 텐서 스케일을 사용합니다. 동적 활성화 스케일링입니다. | rfdetr, birefnet, feynobg | 불필요 |
| `mxfp4` | `Linear`의 OCP MXFP4입니다. E2M1 요소, 32요소 블록, 2의 거듭제곱 E8M0 블록 스케일을 사용합니다. 동적 활성화 스케일링입니다. | rfdetr, birefnet, feynobg | 불필요 |
| `int2` | 연구 전용입니다. 그룹 64의 2비트 가중치와 INT8 활성화를 `Linear`에서 사용합니다. 학습 후 적용만으로는 사용할 수 없으므로 QAT 또는 QAD가 필수입니다. | rfdetr | 필수 |

8비트 미만 레시피는 `nn.Linear`를 대상으로 하며 의도적으로 `yolo9`에서는
거부됩니다. 현재 하드웨어에서 해당 가속은 GEMM에만 적용되므로 컨볼루션은 더 높은
정밀도로 유지됩니다. YOLO9은 `int8` 또는 `fp8`을 사용합니다. `int2`는 레시피에
필요한 QAT 복구를 사용할 수 없는 추론 전용 계열인 `birefnet`과 `feynobg`에서
거부됩니다.

계열별 기본값은 첫 레이어와 헤드를 부동소수점으로 유지하며 YOLO9 DFL 컨볼루션은
절대 양자화하지 않습니다. 고정 적분 기댓값 연산자이기 때문입니다. 이유가 있다면
`keep_high_precision=("head.",)`으로 재정의합니다.

## 보정 데이터와 학습 데이터의 구분

`calib=`은 이미지 수백 장을 받고 레이블을 읽지 않으며 순전파만 실행하여 활성화
범위를 추정합니다. `train()` 및 `val()`의 `data=`는 그래디언트와 메트릭에 사용하는
레이블 데이터셋입니다. 목적이 다른 별도 인수이며 `calib` 기본값은
`coco128.yaml`입니다.

`algorithm="minmax"`는 보정 배치에서 관찰된 절대 극값을 유지하며 `"auto"`가 이를
선택합니다. `"percentile"`은 배치별 0.1 및 99.9 백분위수의 평균을 사용합니다.
트랜스포머 활성화 이상치가 중요한 역할을 하므로 측정 결과 DETR 계열 정확도가
붕괴했습니다. 소형 모델 INT8 민감도를 실제로 해결하는 방법은 충분한 배치로
보정하는 것입니다. `coco128` 기본값을 사용하면 YOLO9-t는 부동소수점 점수에서 약
1 mAP 포인트 이내에 도달합니다. 선택한 알고리즘은 체크포인트 매니페스트에
기록됩니다.

## 정확도 복구

<code-tabs name="train" />

양자화 모듈은 fp32 마스터 가중치를 유지하고 straight-through estimator를 사용한
가짜 양자화를 적용하므로 그래디언트가 마스터에 도달하며 기존 트레이너가 변경 없이
작동합니다. EMA, AMP, 체크포인트 재개, 증류 인수를 모두 함께 사용할 수 있습니다.

QAT는 이미 학습된 모델의 파인튜닝입니다. 처음부터 학습하는 기본값 대신 파인튜닝
학습률을 사용해야 하며, 그렇지 않으면 짧은 실행도 양자화 여부와 무관하게 사전 학습
가중치를 훼손합니다. QAD 사용 가능 여부는 계열의 증류 지원을 따르며 현재는 `yolo9`과
`rfdetr`입니다.

`fp16` 및 `bf16` 양자화 모델은 추론 전용이며 트레이너는 `amp=True` 안내와 함께
거부합니다.

## 내보내기

<code-tabs name="export" />

`format="pt"`는 모델을 확정합니다. 패킹된 저비트 가중치와 스케일이 마스터를
대체하고, `remainder="fp32"`를 전달하지 않으면 양자화되지 않은 나머지를 fp16으로
변환합니다. 패킹 불변 조건은 언패킹 결과가 확정한 디바이스에서 시뮬레이션과 비트
단위로 동일하다는 것입니다. 따라서 확정 파일의 점수는 검증한 모델과 정확히
같습니다. 측정 결과 YOLO9-s int8은 29.5 MB에서 9.6 MB로, RF-DETR-n nvfp4는
122 MB에서 26 MB로 줄어듭니다. 이를 불러오면 추론 준비 모델을 얻으며 `train()`을
호출하면 패킹된 가중치에서 마스터를 자동으로 다시 구성합니다.

`format="onnx"`는 `int8` 모델에 적용되며 모델 자체의 보정 또는 QAT 학습 스케일을
담은 QDQ 그래프를 생성합니다. ONNX Runtime과 TensorRT는 실제 INT8 커널로 이를
실행합니다. 이는 부동소수점 모델에서 ONNX Runtime이 자체적으로 스케일을 계산하는
[`export(format="onnx", int8=True)`](/docs/export/onnx)와 다른 경로입니다.

변환 레시피에는 양자화 전용 내보내기가 전혀 필요하지 않습니다.

<code-tabs name="dequantize" />

## 제약 조건

양자화 산술은 시뮬레이션으로 실행되며 AMP에서도 float32 영역에서 계산되는 가짜
양자화입니다. 시뮬레이션은 수치적으로 정확하므로 어떤 디바이스에서든 `val()` 점수는
양자화 산술에 대한 실제 근거입니다. 속도에 관한 근거는 아닙니다.

두 가지 예외는 네이티브로 실행됩니다. `fp16`과 `bf16`은 일반적인 변환입니다.
확정된 `fp8` 모듈은 Ada, Hopper, Blackwell 계열 하드웨어에서 `torch._scaled_mm`을
통해 패킹된 E4M3 가중치로 GEMM을 직접 실행하며 시뮬레이션과 같은 보정 활성화
스케일을 사용합니다. `LIBREYOLO_KERNELS=off`를 설정하면 모든 환경에서 정확한
시뮬레이션 경로로 복원합니다.

배포 적용 범위는 레시피 목록보다 좁습니다. 여기서는 `int8`만 배포 가능한 ONNX
형식이 있습니다. `fp8` 및 8비트 미만 Linear 레시피는 PyTorch에서 실행하고
`format="pt"`로 확정합니다. 이들 모델에서 ONNX 내보내기를 요청하면 해당 지침과 함께
오류가 발생하며, `int8` 모델에서 ONNX 이외의 형식을 요청해도 마찬가지입니다. 대신
QDQ 그래프에서 다운스트림 엔진을 빌드합니다.

활성화를 보정하지 않은 `int8` 모델을 내보내면 경고를 기록하고 가중치 양자화만 담은
그래프를 생성합니다.

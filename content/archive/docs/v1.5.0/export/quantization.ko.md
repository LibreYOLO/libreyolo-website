---
title: 양자화
seo_title: PyTorch에서 LibreYOLO 모델을 양자화하기
description: >-
  LibreYOLO의 PyTorch 양자화 API: 아홉 가지 레시피, 학습 데이터와 분리된 보정, QAT와 QAD, 그리고 두 가지 배포
  아티팩트.
lead: >-
  LibreYOLO에서의 양자화는 완전히 PyTorch에서 실행됩니다: model.quantize()는 모델의 Conv2d 및 Linear
  모듈을 양자화된 동등 모듈로 교체하고 이를 보정합니다. 결과는 일반적인 predict, val, train 및 save 계약을 유지하므로,
  양자화된 모델은 부동 소수점 모델과 동일한 검증기로 평가됩니다.
keywords:
  - libreyolo 양자화
  - int8 양자화(PTQ)
  - 양자화 인지 학습
  - qat qad
  - nvfp4 mxfp4
  - fp8 e4m3
  - 보정 데이터셋
  - qdq onnx 내보내기
last_verified: 1.5.0
meta:
  - label: 전화하다
    value: 'model.quantize(recipe="int8", calib="coco128.yaml")'
    mono: true
  - label: 명령
    value: libreyolo quantize --model M.pt --recipe int8 --calib coco128.yaml
    mono: true
  - label: 추가
    value: 없음. 양자화는 파이토치에서 실행됩니다.
  - label: 계열들
    value: 'yolo9, rfdetr, birefnet, feynobg'
  - label: 레시피
    value: 'fp16, bf16, fp8, int8, w4a16, w4a8, nvfp4, mxfp4, int2'
    mono: true
  - label: 배포 아티팩트
    value: >-
      export(format="pt") for a packed checkpoint, export(format="onnx") for a
      QDQ INT8 graph
    mono: true
verification: >-
  개발 브랜치에서 libreyolo/quant/api.py, libreyolo/models/base/model.py,
  libreyolo/cli/commands/quantize.py 및 docs/quantization.md를 읽으십시오. 체크포인트 크기 수치는
  docs/quantization.md.에 기록된 측정값입니다.
snippets:
  quantize:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # 구조 교체 및 보정. calib는 작은 레이블 없는 이미지 세트입니다,

        # 활성 범위와 스케일을 도출하기 위해 순방향 전용으로 읽습니다.

        qmodel = model.quantize(recipe="int8", calib="coco128.yaml",
        samples=128)


        print(qmodel.quant_info())

        qmodel.val(data="coco8.yaml")          # 부동 소수점 모델과 동일한 검증기

        qmodel.save("LibreYOLO9s-int8.pt")     # 체크포인트가 양자 매니페스트를 포함하고 있습니다
    - label: CLI
      language: bash
      code: >
        libreyolo quantize --model LibreYOLO9s.pt --recipe int8 --calib
        coco128.yaml
    - label: 논쟁
      language: python
      code: |
        model.quantize(
            recipe="int8",
            calib="coco128.yaml",      # data.yaml 경로나 내장 이름; None은 보정 건너뛰기
            samples=128,               # 최대 보정 이미지
            batch=8,                   # 보정 배치 크기
            algorithm="auto",          # auto와 minmax는 동일하며, percentile은 대안이다
            keep_high_precision=None,  # 아무도 계열 정책을 사용하지 않는다
            verbose=True,
        )
  reload:
    - label: 양자화된 체크포인트가 하나로 다시 로드됩니다
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 퀀트 매니페스트는 양자화된 구조를 재구성하고 크기를 조정합니다
        # 가중치가 로드되기 전에.
        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")
        print(qmodel.quant_info())
  train:
    - label: QAT는 양자화된 모델에서 단순한 train()입니다
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        # 처음부터 하는 실행이 아닌 파인튜닝: 파인튜닝 학습률을 사용하십시오.
        qmodel.train(data="coco8.yaml", epochs=5, lr0=1e-4)
    - label: QAD는 기존의 증류 인수를 추가합니다
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

        # Writes LibreYOLO9s-int8-final.pt: 낮은 비트 가중치와 스케일을 패킹함,
        # FP32 마스터는 제거되고, 비양자화된 나머지는 FP16으로 변환됩니다.
        qmodel.export(format="pt")

        # remainder="fp32"는 비양자화된 텐서를 정확하게 유지합니다.
        qmodel.export(format="pt", remainder="fp32")
    - label: QDQ INT8 ONNX
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        # 그래프 내 QuantizeLinear/DequantizeLinear 쌍이 모델을 운반하는
        # 자체 교정된 또는 QAT로 학습된 스케일.
        qmodel.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9s-int8.pt --format onnx
  dequantize:
    - label: 부동으로 돌아가서 QAT로 학습된 가중치를 유지하기
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")
        qmodel.dequantize()

        # 이제 어떤 부동 소수점 내보내기 도구라도 지원하는 모든 정밀도로 적용됩니다.
        qmodel.export(format="tensorrt", half=True)
source_hash: 4ffb06b87cad017e
---

## 설치

양자화에는 추가 사항이 필요하지 않습니다. 모듈 교체, 보정 패스, 그리고 시뮬레이션된 산술 연산은 모두 PyTorch에서 실행되므로 `pip install libreyolo`가 전체 요구 사항입니다. 배포 산출물은 각자의 형식에서 필요한 것을 요구하며, ONNX 경로의 경우 `libreyolo[onnx]`입니다.

## 양자화

<code-tabs name="quantize" />

`quantize()`는 로드된 모델을 제자리에 변환하고 반환합니다. 그래디언트는 관련되지 않습니다: 교체는 양자화된 모듈을 설치하고 보정 과정은 순전파만 실행합니다.

결과 체크포인트는 `quant` 매니페스트가 첨부된 일반 LibreYOLO 체크포인트이므로 구조와 스케일이 그대로 유지된 채로 다시 로드됩니다:

<code-tabs name="reload" />

QAT 실행 중에 작성된 트레이너 체크포인트는 매니페스트도 포함하고 있어, 이는 이러한 실행에서의 `best.pt` 자체가 양자화된 체크포인트임을 의미합니다.

## 레시피

네 개의 계열가 지원됩니다: `yolo9`, `rfdetr`, `birefnet` 및 `feynobg`.

| 레시피 | 그것이 하는 일 | 계열들 | 교정 |
|---|---|---|---|
| `fp16` | float32 입력 및 출력 계약으로 반정밀도로 캐스팅합니다. 추론 전용입니다. | 네 개 모두 | 없음 |
| `bf16` | float32의 지수 범위를 유지하는 bfloat16으로 캐스트합니다. DETR 스타일 모델에서 fp16이 오버플로우될 때의 수정 방법입니다. 추론 전용입니다. | 네 개 모두 | 없음 |
| `fp8` | `Conv2d` 및 `Linear`의 E4M3 가중치 및 활성화: 채널별 가중치 스케일, 텐서별 활성화 스케일로 보정됨. | 네 개 모두 | 필수 |
| `int8` | `Conv2d` 및 `Linear`에서 W8A8: 채널별 대칭 가중치, 텐서별 아핀 활성화. | 네 개 모두 | 필수, 또는 무게만 해당되는 경우 `calib=None` |
| `w4a16` | 그룹화된 대칭 INT4 가중치, `in_features`를 따라 128 그룹, 부동 소수점 활성화, `Linear`에서. | rfdetr, birefnet, feynobg | 필요하지 않음 |
| `w4a8` | `Linear`에서 그룹화된 INT4 가중치와 보정된 INT8 활성화. | rfdetr, birefnet, feynobg | 필수 |
| `nvfp4` | `Linear`의 W4A4 NVFP4: E2M1 요소, 16요소 블록, FP8 E4M3 블록 스케일, FP32 텐서 스케일. 동적 활성화 스케일링. | rfdetr, birefnet, feynobg | 필요하지 않음 |
| `mxfp4` | `Linear`의 OCP MXFP4: E2M1 요소, 32요소 블록, 2의 거듭제곱 E8M0 블록 스케일. 동적 활성화 스케일링. | rfdetr, birefnet, feynobg | 필요하지 않음 |
| `int2` | 연구용: 그룹화된 2비트 가중치, 그룹 64, 그리고 INT8 활성화, `Linear`에서. 사후 학습만으로는 사용할 수 없으므로 QAT 또는 QAD가 필요합니다. | rfdetr | 필수 |

8비트 미만 레시피는 `nn.Linear`를 대상으로 하고 `yolo9`에서는 의도적으로 거부됩니다: 현재 하드웨어에서는 그 가속이 GEMM만 지원되므로, 컨볼루션은 더 높은 정밀도로 유지됩니다. YOLO9는 `int8` 또는 `fp8`를 사용합니다. `int2`는 `birefnet` 및 `feynobg`에서는 거부되는데, 이들 계열은 추론 전용이므로 레시피를 개선하는 QAT는 거기서 사용할 수 없기 때문입니다.

계열별 기본값은 첫 번째 레이어와 헤드를 부동 소수점으로 유지하며, YOLO9 DFL 컨볼루션은 절대 양자화되지 않습니다: 이는 고정된 적분-기댓값 연산자입니다. 이유가 있는 경우 `keep_high_precision=("head.",)`로 덮어쓸 수 있습니다.

## 보정 데이터는 학습 데이터가 아니다

`calib=`는 수백 장의 이미지를 가져오고, 레이블을 읽지 않고, 활성화 범위를 추정하기 위해 순전파만 수행합니다. `data=`는 `train()`와 `val()`에서 그래디언트와 지표에 사용되는 레이블이 있는 데이터셋입니다. 이들은 목적이 다른 서로 다른 인수이며, `calib`의 기본값은 `coco128.yaml`입니다.

`algorithm="minmax"`는 교정 배치에서 관찰된 절대 극값을 유지하며, 이는 `"auto"`가 선택하는 값입니다. `"percentile"`는 배치별 0.1 및 99.9 백분위수의 평균을 사용합니다. 이는 트랜스포머 활성화 이상치가 구조적 하중을 가지기 때문에 DETR 계열 정확도를 붕괴시키는 것으로 측정되었습니다. 실제로 작은 모델의 INT8 민감도를 해결하는 방법은 충분한 배치로 교정하는 것입니다. `coco128` 기본 설정으로 YOLO9-t는 부동 소수점 점수와 약 1 mAP 포인트 내에 도달합니다. 선택된 알고리즘은 체크포인트 매니페스트에 기록됩니다.

## 정확도를 회복

<code-tabs name="train" />

양자화된 모듈은 fp32 마스터 가중치를 유지하고 스레이트-스루 추정기를 사용하여 가짜 양자화를 적용하므로, 그래디언트가 마스터에 도달하고 기존 학습기들이 변경 없이 작동합니다: EMA, AMP, 체크포인트 재개 및 디스틸레이션 인자들이 모두 조합됩니다.

QAT는 이미 학습된 모델의 파인튜닝입니다. 초기부터 학습하는 기본값 대신 파인튜닝 학습률을 사용하십시오. 그렇지 않으면 짧은 학습만으로도 양자화와 상관없이 사전 학습된 가중치가 손상됩니다. QAD 사용 가능 여부는 계열 증류 지원을 따르며, 오늘날로서는 `yolo9` 및 `rfdetr`를 의미합니다.

`fp16`- 및 `bf16`-양자화 모델은 추론 전용이며, 트레이너는 `amp=True`를 가리키며 이를 거부합니다.

## 내보내기

<code-tabs name="export" />

`format="pt"`는 모델을 결정화합니다. 패킹된 저비트 가중치와 스케일이 마스터를 대체하며, 비양자화된 나머지는 `remainder="fp32"`가 전달되지 않는 한 fp16으로 변환됩니다. 패킹 불변 조건은 언패킹 시 최종화한 장치에서 시뮬레이션을 비트 단위로 재현한다는 것이므로 최종 파일은 검증한 대로 정확한 점수를 기록합니다. 측정 결과: YOLO9-s int8은 29.5MB에서 9.6MB로, RF-DETR-n nvfp4는 122MB에서 26MB로 감소했습니다. 하나를 로드하면 추론 준비가 완료된 모델이 되며, 그 위에서 `train()`를 호출하면 패킹된 가중치에서 마스터를 자동으로 복원합니다.

`format="onnx"`는 `int8` 모델에 적용되며, 모델 자체의 보정된 스케일 또는 QAT 학습된 스케일을 담은 QDQ 그래프를 출력합니다. 이 그래프는 ONNX Runtime과 TensorRT에서 실제 INT8 커널로 실행됩니다. 이는 부동 소수점 모델에서 ONNX Runtime이 스케일을 스스로 도출하는 [`export(format="onnx", int8=True)`](/docs/export/onnx)과는 다른 경로입니다.

캐스트 레시피에는 전혀 양자화된 내보내기가 필요하지 않습니다:

<code-tabs name="dequantize" />

## 제약

양자화 산술은 시뮬레이션에서 실행되며, 이는 AMP 하에서도 float32 영역에서 계산된 가짜 양자화입니다. 시뮬레이션은 수치적으로 정확하므로, 어떤 장치에서든 `val()` 점수는 양자화 산술에 대한 실제 주장입니다. 이는 속도와 관련된 주장이 아닙니다.

두 가지 예외가 네이티브로 실행됩니다. `fp16`와 `bf16`는 일반적인 캐스트입니다. 최종화된 `fp8` 모듈은 Ada, Hopper 및 Blackwell 클래스 하드웨어에서 `torch._scaled_mm`를 통해 패킹된 E4M3 가중치로 직접 GEMM을 실행하며, 시뮬레이션과 동일하게 보정된 활성화 스케일을 사용합니다; `LIBREYOLO_KERNELS=off`를 설정하면 모든 곳에서 정확히 시뮬레이션된 경로가 복원됩니다.

배포 범위는 레시피 목록보다 좁습니다. 여기에서는 `int8`만 배포 가능한 ONNX 형태를 가지고 있습니다; `fp8`와 8비트 미만 선형 레시피는 PyTorch에서 실행되며 `format="pt"`를 통해 결정화됩니다. 이들로부터 ONNX 내보내기를 요청하면 해당 지시어로 오류가 발생하며, `int8` 모델에서 non-ONNX 형식을 요청해도 마찬가지입니다: 대신 QDQ 그래프에서 다운스트림 엔진을 빌드하십시오.

활성화가 한 번도 보정되지 않은 `int8` 모델을 내보내면 경고를 기록하고 가중치 양자화만 포함하는 그래프를 생성합니다.

---
title: NAFNet
families:
  - nafnet
seo_title: 'NAFNet: MIT로 노이즈 제거, 학습, 내보내기'
description: >-
  LibreYOLO에서 이미지 노이즈 제거와 복원에 NAFNet을 사용합니다. MIT가 적용된 SIDD 체크포인트를 설치하고 예측, 학습,
  검증, 내보내기합니다.
lead: >-
  NAFNet은 일반적인 UNet 블록에서 비선형 활성화 함수를 제거하고 요소별 곱셈으로 대체한 이미지 복원용 컨볼루션 네트워크입니다.
  LibreYOLO는 SIDD로 학습한 실제 이미지 노이즈 제거 체크포인트와 함께 복원 한 작업에서 이를 지원합니다.
keywords:
  - NAFNet 사용법
  - 이미지 복원
  - 이미지 노이즈 제거
  - 이미지 디블러링
  - 비선형 활성화 없는 네트워크
  - SIDD
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model("noisy.jpg", save=True)

        restored = result.restored
        print(restored.array.shape)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreNAFNetl-restore-sidd.pt source=noisy.jpg
        save=True
    - label: 복원된 이미지 저장
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model.predict("noisy.jpg")

        result.restored.save("denoised.png")
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        model.train(data="my-dataset.yaml", epochs=100, imgsz=256, batch=16,
        lr0=1e-3)
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
        \
          epochs=100 imgsz=256 batch=16 lr0=1e-3
    - label: 체크포인트 출처
      language: python
      code: |
        from libreyolo import LibreYOLO

        # degradation과 dataset은 저장된 체크포인트에 기록되며
        # 학습되는 내용은 바꾸지 않습니다.
        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            degradation="denoise",
            dataset="MyDataset",
        )
    - label: 다중 GPU
      language: bash
      code: >
        libreyolo train model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
        \
          epochs=100 device=0,1 batch=32
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # val()은 객체가 아니라 일반 dict를 반환합니다
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.export(format="onnx", imgsz=256)
        model.export(format="tensorrt", imgsz=256, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreNAFNetl-restore-sidd.pt format=onnx
        imgsz=256

        libreyolo export model=LibreNAFNetl-restore-sidd.pt format=tensorrt
        imgsz=256 half=True
    - label: 내보낸 파일 사용하기
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 팩토리는 파일 접미사로 라우팅하므로 내보낸 아티팩트도
        # 체크포인트처럼 불러오며 동일한 Results 객체를 반환합니다.
        model = LibreYOLO("LibreNAFNetl-restore-sidd.onnx")
        result = model("noisy.jpg")

        result.restored.save("denoised.png")
source_hash: 9bae9f82bee741bf
---

## 설치

NAFNet에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에
포함됩니다.

```bash
pip install libreyolo
```

## 예측

가중치는 처음 사용할 때 Hugging Face에서 내려받아 로컬에 캐시됩니다.

<code-tabs name="predict" />

반환되는 `Results` 객체에는 이 계열용 필드 `restored` 하나가 들어 있습니다. 원본
캔버스의 밀집 HWC uint8 RGB 이미지이며 순회할 바운딩 박스는 없습니다. `save=True`는
입력 위에 어노테이션을 그리는 대신 복원된 이미지를 디스크에 바로 씁니다. 복원은
필터링할 탐지를 생성하지 않으므로 다른 모든 계열과의 시그니처 동등성을 위해
`conf`, `iou`, `max_det`를 허용하지만 효과가 없습니다. 소스, 스트리밍, 결과 처리는
[예측](/docs/predict)을 참조합니다.

## 변형

너비는 32인 `s`와 64인 `l` 두 가지이며 둘 다 256 px 학습 패치를 중심으로
구성됩니다. 크기와 관계없이 예측과 검증은 네이티브 이미지 해상도에서 실행하며
네트워크 다운샘플 배수에 맞추기 위한 패딩만 적용합니다. 현재 공개된 크기는 SIDD로
학습한 실제 이미지 노이즈 제거 체크포인트인 `l`뿐입니다.

## 학습

NAFNet은 자체적으로 짝을 이룬 열화 및 깨끗한 이미지에서 파인튜닝합니다. 데이터셋
YAML은 열화 이미지가 있는 `inputs/<split>/` 폴더와 파일 이름 줄기가 일치하는 깨끗한
대상이 있는 `targets/<split>/` 폴더를 가리킵니다. `degradation`과 `dataset`은
출처를 위해 저장된 체크포인트에 기록하는 선택적 문자열이며 학습에는 관여하지
않습니다.

<code-tabs name="train" />

다른 값을 지정하지 않으면 트레이너는 AdamW, `lr0=1e-3`, 배치 16, 256 px 크롭으로
100 에폭을 실행하며 PSNR 개선 없이 50 에폭이 지나면 조기 종료합니다. 이 계열에는
LoRA 경로가 없습니다. `NAFNetTrainer`가 어댑터 파인튜닝을 선택하지 않으므로
`lora=True`는 실행하지 않고 오류를 발생시킵니다.

학습 중 네트워크는 일반 전역 평균 풀링으로 실행합니다. NAFNet의 추론 전용 윈도우
로컬 풀링(Test-time Local Converter)은 첫 에폭 전에 분리하고 학습이 끝나면 다시
연결합니다. 고정 윈도우 로컬 풀을 통한 역전파는 체크포인트의 추론 사용 방식과
일치하지 않기 때문입니다.

데이터셋, 증강, 다중 GPU, 로거는 [학습](/docs/train)을 참조합니다.

## 검증

`val()`은 전체 유효 캔버스의 RGB에서 계산한 `metrics/PSNR` 및 `metrics/SSIM`
딕셔너리를 반환합니다. SSIM은 sigma 1.5의 11x11 가우시안 윈도우를 사용하며, 최상
체크포인트 선택용 `fitness`는 PSNR 값입니다. `data`는 학습에 사용한 것과 같은 짝을
이룬 이미지 데이터셋 형식을 가리킵니다.

<code-tabs name="val" />

## 내보내기

<export-matrix />

내보낸 아티팩트는 파일 접미사에 따라 `LibreYOLO()`로 다시 불러오므로 `.onnx` 또는
`.engine` 파일이 체크포인트처럼 동작하며 동일한 `Results`를 반환하고 `restored`에
출력 이미지가 들어 있습니다. NAFNet은 고정 공간 해상도로 내보냅니다. `imgsz`는
네트워크 다운샘플 배수(두 아키텍처 너비 모두 16)로 나누어떨어져야 하며,
`dynamic=True`일 때는 배치 차원만 동적이고 높이와 너비는 내보낼 때 고정됩니다.

<code-tabs name="export" />

## 체크포인트

이 계열에서 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />

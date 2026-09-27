---
title: FCN
families:
  - fcn
seo_title: 'FCN: dự đoán và xuất ResNet FCN theo BSD-3-Clause'
description: >-
  Dùng FCN trong LibreYOLO để phân đoạn ngữ nghĩa. Cài đặt, dự đoán, xác thực và
  xuất các checkpoint FCN ResNet giãn của torchvision.
lead: >-
  Bộ phân loại dense theo từng pixel thay các lớp fully connected của detector
  bằng phép tích chập, vì vậy đầu ra là map lớp đối tượng ở độ phân giải đầy đủ
  thay vì box. LibreYOLO chỉ cung cấp mô hình này cho phân đoạn ngữ nghĩa.
keywords:
  - FCN
  - fully convolutional network
  - phân đoạn ngữ nghĩa
  - dự đoán dense
  - ResNet
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFCNr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) ID lớp đối tượng
        print(mask.classes)      # các ID lớp đối tượng trong ảnh, đã sắp xếp
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFCNr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCNr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFCNr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCNr50.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFCNr50.pt format=onnx
        libreyolo export model=LibreFCNr50.pt format=tensorrt half=True
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến dựa trên hậu tố tệp, vì vậy artifact đã xuất tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreFCNr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 7776b0fc85a208fb
---

## Cài đặt

FCN không cần extra tùy chọn. Mọi thành phần mà mô hình import đều có trong bản cài đặt cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Phân đoạn ngữ nghĩa trả về một ID lớp đối tượng cho mỗi pixel thay vì box, vì vậy `result.semantic_mask` chứa mảng `(H, W)` trong `.data` và danh sách ID lớp đối tượng có trong ảnh ở `.classes`. `conf`, `iou` và `max_det` được chấp nhận để giữ tính tương đồng của API nhưng không có tác dụng: mô hình gán một lớp đối tượng cho mỗi pixel bằng argmax, không có ngưỡng độ tin cậy hay bước NMS. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Có hai độ sâu ResNet, cả hai ở đầu vào cố định 520 px. Graph inference của thư viện là FCN ResNet giãn của torchvision, không phải mạng FCN-8s dựa trên VGG với skip connection từ bài báo gốc.

LibreYOLO không huấn luyện FCN: `train()` phát sinh `NotImplementedError` cho họ mô hình này, được [cấp hỗ trợ](/docs/models) ở trên đánh dấu là chỉ dành cho inference. Hai checkpoint đã phát hành là trọng số được huấn luyện trên COCO của chính torchvision, được chuyển đổi cho loader của LibreYOLO.

## Xác thực

`val()` trả về `metrics/mIoU` và `metrics/pixel_accuracy`, được đo trên mọi dataset có định dạng giống định dạng bạn đã dùng để huấn luyện.

<code-tabs name="val" />

## Xuất

<export-matrix />

Artifact đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp `.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng đối tượng `Results`. Trang [Xuất](/docs/export) liệt kê các đối số mà mọi định dạng chấp nhận.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã phát hành cho họ mô hình này.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>

---
title: DeepLabv3
families:
  - deeplabv3
seo_title: 'DeepLabv3: dự đoán và xuất mô hình phân đoạn ngữ nghĩa ASPP'
description: >-
  Dùng DeepLabv3 trong LibreYOLO để phân đoạn ngữ nghĩa. Cài đặt, dự đoán, xác
  thực và xuất các checkpoint ResNet và MobileNetV3 của torchvision.
lead: >-
  Mạng phân đoạn ngữ nghĩa gộp các đặc trưng song song ở nhiều tỷ lệ dilation
  (atrous spatial pyramid pooling) trước khi phân loại từng pixel. LibreYOLO chỉ
  cung cấp mô hình này cho phân đoạn ngữ nghĩa.
keywords:
  - DeepLabv3
  - atrous spatial pyramid pooling
  - ASPP
  - phân đoạn ngữ nghĩa
  - dự đoán dense
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) ID lớp đối tượng
        print(mask.classes)      # các ID lớp đối tượng trong ảnh, đã sắp xếp
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeepLabv3r50-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeepLabv3r50-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDeepLabv3r50-sem.pt format=onnx

        libreyolo export model=LibreDeepLabv3r50-sem.pt format=tensorrt
        half=True
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến dựa trên hậu tố tệp, vì vậy artifact đã xuất tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreDeepLabv3r50-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 7abf11ebb6cece18
---

## Cài đặt

DeepLabv3 không cần extra tùy chọn. Mọi thành phần mà mô hình import đều có trong bản cài đặt cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ. Hậu tố tên tệp `-sem` là bắt buộc với họ mô hình này.

<code-tabs name="predict" />

Phân đoạn ngữ nghĩa trả về một ID lớp đối tượng cho mỗi pixel thay vì box, vì vậy `result.semantic_mask` chứa mảng `(H, W)` trong `.data` và danh sách ID lớp đối tượng có trong ảnh ở `.classes`. `conf`, `iou` và `max_det` được chấp nhận để giữ tính tương đồng của API nhưng không có tác dụng: mô hình gán một lớp đối tượng cho mỗi pixel bằng argmax, không có ngưỡng độ tin cậy hay bước NMS. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Có ba backbone: ResNet-50 giãn, ResNet-101 giãn và MobileNetV3-Large giãn. Đây là DeepLabv3, không phải DeepLabv3+, vì vậy không có giai đoạn decoder hay tinh chỉnh CRF, phù hợp với cách triển khai của torchvision thay vì mã tham chiếu riêng của bài báo.

LibreYOLO không huấn luyện DeepLabv3: `train()` phát sinh `NotImplementedError` cho họ mô hình này, được [cấp hỗ trợ](/docs/models) ở trên đánh dấu là chỉ dành cho inference. Ba checkpoint đã phát hành là trọng số COCO với nhãn VOC của chính torchvision, được chuyển đổi cho loader của LibreYOLO.

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

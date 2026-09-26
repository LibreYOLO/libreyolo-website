---
title: EfficientDet
families:
  - efficientdet
seo_title: 'EfficientDet: phát hiện đối tượng trong LibreYOLO'
description: >-
  Chạy EfficientDet D0-D4 trong LibreYOLO: các detector BiFPN để dự đoán, xác
  thực và xuất sang ONNX, TensorRT và OpenVINO theo Apache-2.0.
lead: >-
  EfficientDet ghép backbone EfficientNet với mạng kim tự tháp đặc trưng hai
  chiều lặp lại (BiFPN) và cùng scale độ sâu, độ rộng cùng độ phân giải trên năm
  kích thước. LibreYOLO cung cấp mô hình này như một detector chỉ dành cho
  inference.
keywords:
  - EfficientDet
  - BiFPN
  - EfficientNet
  - phát hiện đối tượng
  - compound scaling
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEfficientDetd0.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEfficientDetd0.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientDetd0.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEfficientDetd0.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientDetd0.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEfficientDetd0.pt format=onnx
        libreyolo export model=LibreEfficientDetd0.pt format=tensorrt half=True
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Factory định tuyến dựa trên hậu tố tệp, vì vậy artifact đã xuất tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreEfficientDetd0.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 12c61fb0035437ce
---

## Cài đặt

EfficientDet không cần extra tùy chọn. Mọi thành phần mà mô hình import đều có trong bản cài đặt cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Đối tượng `Results` trả về cũng là đối tượng mà mọi họ mô hình đều trả về, vì vậy việc chuyển sang detector khác chỉ cần thay đổi một dòng. EfficientDet giải mã các candidate dựa trên anchor rồi chạy non-maximum suppression theo lớp đối tượng, vì vậy `conf`, `iou` và `max_det` đều có tác dụng thực sự ở đây. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Có năm kích thước từ D0 đến D4. Mỗi bước tăng kết hợp backbone EfficientNet lớn hơn với BiFPN sâu hơn, rộng hơn và prediction head sâu hơn, vì vậy số lượng tham số và lượng tính toán tăng cùng nhau theo quy tắc compound scaling của bài báo.

## Xác thực

`val()` trả về dictionary gồm các key `metrics/` cho precision, recall, mAP 50 và mAP 50-95, được đo trên mọi dataset có định dạng giống định dạng bạn đã dùng để huấn luyện.

<code-tabs name="val" />

## Xuất

<export-matrix />

Artifact đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp `.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng đối tượng `Results`.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã phát hành cho họ mô hình này.

<checkpoint-table />

## Giấy phép

<provenance-box>

Các checkpoint D0-D4 của LibreYOLO được chuyển đổi thông qua dự án rwightman/efficientdet-pytorch dùng Apache-2.0; bản thân dự án này mirror trọng số được huấn luyện bằng TensorFlow chính thức từ google/automl mà không thay đổi tensor đã học. Không có mã nguồn nào từ dự án zylo117/Yet-Another-EfficientDet-Pytorch dùng LGPL được tham khảo hoặc sử dụng.

</provenance-box>

---
title: LW-DETR
families:
  - lwdetr
seo_title: 'LW-DETR: dự đoán và xuất theo Apache-2.0'
description: >-
  Chạy LW-DETR trong LibreYOLO để phát hiện đối tượng thời gian thực. Cài đặt,
  dự đoán, xác thực và xuất năm kích thước dựa trên ViT, tất cả đều dùng giấy
  phép Apache-2.0.
lead: >-
  Detection transformer plain-ViT được Baidu định vị là giải pháp thay thế thời
  gian thực cho các detector YOLO. LibreYOLO cung cấp năm kích thước chỉ để
  inference phát hiện.
keywords:
  - LW-DETR
  - detection transformer
  - phát hiện đối tượng thời gian thực
  - plain ViT
  - DETR
  - Baidu
  - Atten4Vis
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLWDETRt.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreLWDETRt.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLWDETRt.pt")

        # val() trả về dict thuần túy, không phải đối tượng
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreLWDETRt.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLWDETRt.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreLWDETRt.pt format=onnx imgsz=640

        libreyolo export model=LibreLWDETRt.pt format=tensorrt imgsz=640
        half=True
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến dựa trên hậu tố tệp, vì vậy artifact đã xuất tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreLWDETRt.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: badd1d8255df5bbd
---

## Cài đặt

LW-DETR không cần extra tùy chọn. Mọi thành phần mà mô hình import đều có trong bản cài đặt cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Đối tượng `Results` trả về cũng là đối tượng mà mọi họ mô hình đều trả về, vì vậy việc chuyển sang detector khác chỉ cần thay đổi một dòng. `conf` và `max_det` lọc lựa chọn query; `iou` được chấp nhận để giữ tính tương đồng của API nhưng không có tác dụng vì decoder là bộ dự đoán tập hợp không có bước NMS. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

LW-DETR chỉ dành cho inference trong LibreYOLO. Upstream huấn luyện bằng giám sát one-to-many Group-DETR trên nhiều nhóm query và classification loss nhận biết IoU; công thức đó chưa được kết nối ở đây, vì vậy `train()` phát sinh `NotImplementedError`.

## Biến thể

Có năm kích thước, tất cả dùng chung encoder plain-ViT, projector đa tỷ lệ và decoder deformable DETR, đồng thời đều chạy ở cùng độ phân giải đầu vào. Hai kích thước nhỏ nhất dùng chung độ rộng encoder và khác nhau theo độ sâu block; hai kích thước tiếp theo dùng chung encoder rộng hơn và khác nhau theo số cấp projector đưa vào decoder; kích thước lớn nhất dùng encoder rộng nhất.

## Xác thực

`val()` trả về dictionary gồm các key `metrics/` cho precision, recall, mAP 50 và mAP 50-95, được đo trên mọi dataset có định dạng giống định dạng bạn đã dùng để huấn luyện.

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

## Trích dẫn

<citation-block />



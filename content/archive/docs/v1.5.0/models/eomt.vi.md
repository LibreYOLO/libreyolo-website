---
title: EoMT
families:
  - eomt
seo_title: 'EoMT: dự đoán phân đoạn ngữ nghĩa, thực thể và toàn cảnh'
description: >-
  Dùng EoMT trong LibreYOLO để phân đoạn ngữ nghĩa, thực thể và toàn cảnh trên
  vision transformer DINOv2 thuần túy, không cần decoder. Dùng giấy phép MIT.
lead: >-
  Mạng phân đoạn được xây dựng trên vision transformer thuần túy không có pixel
  decoder chuyên dụng: các query được học bổ sung vào chính encoder sẽ dự đoán
  mặt nạ. LibreYOLO hỗ trợ mô hình này cho phân đoạn ngữ nghĩa, thực thể và toàn
  cảnh.
keywords:
  - EoMT
  - encoder-only mask transformer
  - DINOv2
  - phân đoạn toàn cảnh
  - phân đoạn thực thể
  - phân đoạn ngữ nghĩa
last_verified: 1.5.0
snippets:
  predict:
    - label: Phân đoạn ngữ nghĩa
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTl-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) ID lớp đối tượng
        print(mask.classes)      # các ID lớp đối tượng trong ảnh, đã sắp xếp
    - label: Phân đoạn thực thể
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Hậu tố -seg trong tên tệp chọn tác vụ thực thể, vì vậy không cần
        # đối số task ở đây.
        model = LibreYOLO("LibreEoMTl-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.boxes.xyxy)
        print(result.masks.data.shape)
    - label: Phân đoạn toàn cảnh
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        result = model(SAMPLE_IMAGE, save=True)

        pan = result.panoptic
        print(pan.data.shape)       # (H, W) ID segment
        print(pan.segments_info)    # [{"id": ..., "category_id": ...}, ...]
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEoMTl-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Semantic
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: Instance segmentation
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # mặt nạ
        print(metrics["metrics/mAP50-95(B)"])   # box
    - label: Panoptic
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PQ"])
        print(metrics["metrics/SQ"], metrics["metrics/RQ"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEoMTl-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEoMTl-sem.pt format=onnx
        libreyolo export model=LibreEoMTl-sem.pt format=tensorrt half=True
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến dựa trên hậu tố tệp, vì vậy artifact đã xuất tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreEoMTl-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 64b2da642999f150
---

## Cài đặt

EoMT không cần extra tùy chọn. Mọi thành phần mà mô hình import đều có trong bản cài đặt cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ. Hậu tố tác vụ trong tên tệp (`-sem`, `-seg`, `-panoptic`) chọn tác vụ, còn `LibreYOLO()` suy ra tác vụ từ tên tệp đó nên không cần đối số `task=`.

<code-tabs name="predict" />

Phân đoạn ngữ nghĩa điền `result.semantic_mask`, một mảng ID lớp đối tượng `(H, W)` trong `.data`. Phân đoạn thực thể điền `result.boxes` và `result.masks`, cùng shape mà mọi họ phân đoạn khác trả về. Phân đoạn toàn cảnh điền `result.panoptic`: map ID segment `(H, W)` trong `.data`, cùng `.segments_info`, danh sách dict `{"id", "category_id"}`, mỗi segment một dict. `conf` lọc lựa chọn query; `iou` không có tác dụng với tác vụ ngữ nghĩa vì tác vụ dùng argmax theo từng pixel mà không có bước NMS. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Có ba kích thước encoder s/b/l, tất cả dựa trên DINOv2. Checkpoint ngữ nghĩa được huấn luyện trên ADE20K ở 512 px; checkpoint thực thể và toàn cảnh được huấn luyện trên COCO ở 640 px, cùng một checkpoint thực thể thứ hai được huấn luyện ở 1280 px. Upstream chỉ cung cấp trọng số phân đoạn thực thể DINOv2 ở kích thước l; s và b chỉ được phát hành cho tác vụ ngữ nghĩa và toàn cảnh. Các biến thể EoMT dựa trên DINOv3 có ở upstream nhưng không được cung cấp ở đây vì phụ thuộc vào trọng số DINOv3 phi thương mại có kiểm soát truy cập.

LibreYOLO không huấn luyện EoMT: `train()` phát sinh `NotImplementedError` cho họ mô hình này, được [cấp hỗ trợ](/docs/models) ở trên đánh dấu là chỉ dành cho inference.

## Xác thực

`val()` điều phối theo tác vụ. Tác vụ ngữ nghĩa trả về `metrics/mIoU` và `metrics/pixel_accuracy`. Phân đoạn thực thể trả về cùng các key mAP mặt nạ và box như những họ phân đoạn khác. Tác vụ toàn cảnh trả về Panoptic Quality dưới dạng `metrics/PQ`, được tách thành `metrics/SQ` (chất lượng phân đoạn) và `metrics/RQ` (chất lượng nhận dạng), cùng `metrics/PQ_things` và `metrics/PQ_stuff`.

<code-tabs name="val" />

## Xuất

<export-matrix />

Hiện chỉ tác vụ ngữ nghĩa có thể xuất: phân đoạn thực thể và toàn cảnh gọi `export()` sẽ nhận `NotImplementedError` vì đầu ra query-mask của chúng chưa có hợp đồng xuất runtime. Artifact ngữ nghĩa đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp `.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng đối tượng `Results`.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã phát hành cho họ mô hình này.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />



---
title: CenterNet
families:
  - centernet
seo_title: 'CenterNet: phát hiện đối tượng trong LibreYOLO'
description: >-
  Chạy CenterNet (Objects as Points) trong LibreYOLO với backbone ResDCN-18 và
  DLA-34. Dự đoán, kiểm định và xuất sang ONNX theo giấy phép MIT. Không có
  luồng huấn luyện.
lead: >-
  CenterNet mô hình hóa một đối tượng bằng điểm tâm của bounding box và hồi quy
  mọi thuộc tính còn lại từ đỉnh heatmap, nên nó không cần anchor và không cần
  bước non-maximum-suppression. LibreYOLO cung cấp nó như một bộ phát hiện chỉ
  dùng cho suy luận (inference).
keywords:
  - CenterNet
  - Objects as Points
  - keypoint detection
  - anchor-free detector
  - phát hiện đối tượng không cần anchor
  - ResDCN-18
  - DLA-34
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCenterNetresdcn18.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreCenterNetresdcn18.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: DLA-34
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCenterNetdla34.pt")
        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCenterNetresdcn18.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreCenterNetresdcn18.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCenterNetresdcn18.pt")

        # Xuất ONNX cần opset 16 trở lên: tầng upsampling deformable-convolution
        # được hạ xuống thành GridSample, toán tử mà opset 16 mới giới thiệu
        model.export(format="onnx", opset=18)
        model.export(format="tensorrt")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreCenterNetresdcn18.pt format=onnx opset=18
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Factory định tuyến theo phần mở rộng của tệp, nên một artifact đã xuất
        # được nạp như mọi checkpoint và trả về cùng một đối tượng Results
        model = LibreYOLO("LibreCenterNetresdcn18.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 20aaef83cc95590d
---

## Cài đặt

CenterNet không cần extra tùy chọn nào. Mọi thứ nó import đều nằm trong bản cài
đặt cơ bản.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải về từ Hugging Face ở lần dùng đầu tiên và được lưu cache cục
bộ.

<code-tabs name="predict" />

Đối tượng `Results` trả về chính là đối tượng mà mọi family đều trả về, nên đổi
sang một bộ phát hiện khác chỉ là thay đổi một dòng. `conf` và `max_det` lọc các
đỉnh heatmap đã được xếp hạng; `iou` vẫn được chấp nhận để đồng nhất API nhưng
không có tác dụng, vì bước decode top-k đỉnh của CenterNet không cần bước triệt
tiêu dựa trên IoU giữa các box. Xem [dự đoán](/docs/predict) để biết về nguồn
đầu vào, luồng (stream) và cách xử lý kết quả.

## Biến thể

Hai backbone. `resdcn18` ghép trunk ResNet-18 với upsampling
deformable-convolution; `dla34` ghép trunk DLA-34 với upsampling iterative
deep-aggregation. Cả hai cùng đưa dữ liệu vào ba head dense giống nhau (heatmap,
chiều rộng/chiều cao, offset) và cùng một khung ảnh đầu vào.

## Kiểm định

`val()` trả về một dictionary gồm các khóa `metrics/` bao trùm precision, recall,
mAP 50 và mAP 50-95, được đo trên bất kỳ tập dữ liệu (dataset) nào ở định dạng
mà bạn đã huấn luyện.

<code-tabs name="val" />

## Xuất mô hình

<export-matrix />

Xuất ONNX yêu cầu opset 16 trở lên: tầng upsampling deformable-convolution
trong cả hai backbone được hạ xuống thành toán tử ONNX `GridSample`, vốn được
opset 16 giới thiệu. Yêu cầu một opset cũ hơn sẽ báo lỗi trước khi bắt đầu
tracing.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã công bố cho family này.

<checkpoint-table />

## Giấy phép

<provenance-box>

Đồ thị ResDCN-18 cũng ghi công human-pose-estimation.pytorch của Microsoft theo
giấy phép MIT, và đồ thị DLA-34 ghi công bản hiện thực DLA theo BSD-3-Clause của
Fisher Yu. LibreYOLO không đóng gói kèm phần mở rộng DCNv2 gốc mà dự án upstream
đã dùng; thay vào đó, thực thi native chạy `deform_conv2d` theo BSD-3-Clause của
torchvision, còn bản hiện thực portable chỉ dùng cho việc xuất mô hình được viết
riêng cho LibreYOLO.

</provenance-box>

## Trích dẫn

<citation-block />

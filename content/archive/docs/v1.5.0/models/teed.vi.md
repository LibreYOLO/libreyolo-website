---
title: TEED
families:
  - teed
seo_title: 'TEED: phát hiện cạnh, dùng checkpoint của bạn'
description: >-
  Dùng TEED trong LibreYOLO để dự đoán xác suất cạnh dense. Chuyển đổi
  checkpoint mà bạn được cấp phép sử dụng, sau đó dự đoán, xác thực và xuất.
lead: >-
  TEED (Tiny and Efficient Edge Detector) là mạng tích chập nhỏ dự đoán map xác
  suất cạnh dense từ một ảnh RGB. LibreYOLO bọc kiến trúc này chỉ cho phát hiện
  cạnh; không có checkpoint nào đi kèm thư viện.
keywords:
  - TEED
  - Tiny and Efficient Edge Detector
  - phát hiện cạnh
  - BIPED
  - dự đoán dense
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreTEEDt-edge.pt")
        result = model(SAMPLE_IMAGE, save=True)

        edges = result.edges
        print(edges.array.shape)        # (H, W) float32 trong [0, 1]
        print(edges.binary(0.5).sum())  # số pixel cạnh sau khi áp ngưỡng
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=weights/LibreTEEDt-edge.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreTEEDt-edge.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=352)

        print(metrics["metrics/ODS"])   # F-measure ở tỷ lệ dataset tối ưu
        print(metrics["metrics/OIS"])   # F-measure ở tỷ lệ ảnh tối ưu
    - label: CLI
      language: bash
      code: >
        libreyolo val model=weights/LibreTEEDt-edge.pt data=my-dataset.yaml
        imgsz=352
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreTEEDt-edge.pt")
        model.export(format="onnx", imgsz=352)
        model.export(format="tensorrt", imgsz=352, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=weights/LibreTEEDt-edge.pt format=onnx imgsz=352

        libreyolo export model=weights/LibreTEEDt-edge.pt format=tensorrt
        imgsz=352 half=True
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreTEEDt-edge.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.edges.array.shape)
source_hash: c7203b254e460258
---

## Cài đặt

TEED không cần extra tùy chọn. Mọi thành phần mà mô hình import đều có trong bản cài đặt cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

LibreYOLO không cung cấp checkpoint TEED. Trọng số được phát hành chính thức được huấn luyện trên BIPED, có điều khoản dataset đã công bố giới hạn việc sử dụng vào mục đích phi thương mại, vì vậy LibreYOLO không mirror chúng. Hãy chuyển đổi checkpoint mà bạn được cấp phép sử dụng bằng `weights/convert_teed_weights.py`; script này kiểm tra các key tensor theo kiến trúc runtime trước khi ghi tệp mà LibreYOLO có thể tải trực tiếp:

```bash
python weights/convert_teed_weights.py upstream.pth weights/LibreTEEDt-edge.pt --verify
```

<code-tabs name="predict" />

`result.edges` chứa kết quả: một mảng float32 `(H, W)` trong `[0, 1]`, với `.binary(threshold)` trả về mặt nạ cạnh kiểu boolean. Không có box, vì vậy `conf`, `iou` và `max_det` không có tác dụng. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

TEED cung cấp một kích thước trong LibreYOLO. Bộ công cụ benchmark của LibreYOLO chưa đo họ mô hình này, vì vậy không có số liệu đã công bố để so sánh.

## Xác thực

`val()` báo cáo F-measure ODS và OIS kiểu BSDS trên dataset cạnh theo cặp: ảnh đặt cạnh các edge map có cùng phần tên gốc, cùng mặt nạ hợp lệ tùy chọn để các pixel padding không bao giờ được tính. `imgsz` phải chia hết cho downsample stride của mạng, LibreYOLO sẽ phát sinh lỗi rõ ràng nếu không đạt.

<code-tabs name="val" />

## Xuất

<export-matrix />

Việc xuất cạnh sử dụng hợp đồng runtime batch 1, độ phân giải cố định: `dynamic` và `batch` khác 1 bị từ chối, graph đã xuất cho đầu ra là một map xác suất hợp nhất duy nhất. Artifact đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp `.onnx` hoạt động như checkpoint và trả về cùng đối tượng `Results`.

<code-tabs name="export" />

## Giấy phép

<provenance-box>

LibreYOLO không phát hành checkpoint TEED. Không có nội dung nào được mirror trong tổ chức LibreYOLO; thay vào đó hãy chuyển đổi checkpoint mà bạn có giấy phép sử dụng bằng `weights/convert_teed_weights.py`.

</provenance-box>

## Trích dẫn

<citation-block />



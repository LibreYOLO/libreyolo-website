---
title: MoGe-2
families:
  - moge2
seo_title: 'MoGe-2: dự đoán, xác thực và xuất pháp tuyến bề mặt'
description: >-
  Dùng MoGe-2 trong LibreYOLO để dự đoán pháp tuyến bề mặt dense. Cài đặt, dự
  đoán, xác thực và xuất các checkpoint ViT-S, ViT-B và ViT-L chính thức.
lead: >-
  MoGe-2 là mô hình hình học đơn ảnh một forward pass, dự đoán trường pháp tuyến
  bề mặt dense từ một ảnh RGB. LibreYOLO chỉ hỗ trợ mô hình để ước lượng pháp
  tuyến qua các checkpoint ViT-S, ViT-B và ViT-L chính thức.
keywords:
  - MoGe-2
  - MoGe 2
  - ước lượng pháp tuyến bề mặt
  - hình học đơn ảnh
  - normal map
  - dự đoán dense
  - DINOv2
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE, save=True)

        normal = result.normal_map
        print(normal.array.shape)   # vector đơn vị float32 (H, W, 3)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMoGe2s-normal.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=518)

        print(metrics["metrics/mean_angular_error"])   # độ
        print(metrics["metrics/median_angular_error"])
        print(metrics["metrics/within_11_25"])          # phần trăm pixel
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMoGe2s-normal.pt data=my-dataset.yaml imgsz=518
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        model.export(format="onnx", imgsz=518)
        model.export(format="tensorrt", imgsz=518, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreMoGe2s-normal.pt format=onnx imgsz=518

        libreyolo export model=LibreMoGe2s-normal.pt format=tensorrt imgsz=518
        half=True
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.normal_map.array.shape)
source_hash: ddfacf6b7e9729f6
---

## Cài đặt

MoGe-2 không cần extra tùy chọn. Mọi thành phần mà mô hình import đều có trong bản cài đặt cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tự động tải trong lần sử dụng đầu tiên: LibreYOLO tìm nạp kích thước tương ứng trực tiếp từ các checkpoint chính thức và lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

MoGe-2 trả về trường dense thay vì tập detection, vì vậy `result.boxes` trống và `conf`, `iou` cùng `max_det` không có tác dụng. `result.normal_map` chứa kết quả: mảng vector đơn vị `(H, W, 3)` trong hệ tọa độ camera OpenCV, trong đó `+x` hướng sang phải, `+y` hướng xuống, `+z` hướng vào cảnh, còn bề mặt hướng về camera có giá trị `(0, 0, -1)`. Dự đoán danh sách ảnh chạy một forward pass cho mỗi ảnh; họ mô hình này không có tuyến nhanh batch xếp chồng. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Ba kích thước encoder được cung cấp dưới dạng checkpoint riêng: ViT-S, ViT-B và ViT-L, tất cả ở cùng độ phân giải đầu vào. Bộ công cụ benchmark của LibreYOLO chưa đo họ mô hình này, vì vậy không có số liệu độ chính xác đã công bố để so sánh; hãy chọn kích thước theo ngân sách tính toán của bạn.

## Xác thực

`val()` đo sai số góc trên dataset normal-map theo cặp: ảnh đặt cạnh PNG pháp tuyến 16 bit có cùng phần tên gốc, cùng mặt nạ hợp lệ tùy chọn để pixel padding và không hợp lệ không bao giờ được tính. Hàm trả về sai số góc trung bình và trung vị theo độ, cùng phần trăm pixel trong phạm vi 11,25, 22,5 và 30 độ.

<code-tabs name="val" />

## Xuất

<export-matrix />

Việc xuất pháp tuyến dùng hợp đồng runtime batch 1, độ phân giải cố định: `dynamic` và `batch` khác 1 bị từ chối, còn `imgsz` phải chia hết cho kích thước patch của encoder ViT, được LibreYOLO kiểm tra trước khi bắt đầu lượt chạy. Artifact đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp `.onnx` hoạt động như checkpoint và trả về cùng đối tượng `Results`.

<code-tabs name="export" />

## Giấy phép

<provenance-box>

LibreYOLO không sao chép các checkpoint này vào tổ chức riêng. `LibreYOLO("LibreMoGe2s-normal.pt")` tải trực tiếp kích thước tương ứng từ các repo Hugging Face chính thức ở revision cố định và xác minh tệp theo checksum SHA-256 đã ghi trước khi sử dụng.

</provenance-box>

## Trích dẫn

<citation-block />



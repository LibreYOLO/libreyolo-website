---
title: Depth Anything 3
families:
  - depth_anything3
seo_title: 'Depth Anything 3: dự đoán độ sâu đơn ảnh trong LibreYOLO'
description: >-
  Dùng Depth Anything 3 trong LibreYOLO để ước lượng độ sâu đơn ảnh. Cài đặt, dự
  đoán, xác thực và xuất checkpoint DA3MONO-LARGE theo Apache-2.0.
lead: >-
  Depth Anything 3 là DINOv2 transformer thuần túy được huấn luyện để dự đoán độ
  sâu và hình học camera từ một hoặc nhiều góc nhìn mà không chuyên biệt hóa
  kiến trúc. LibreYOLO port checkpoint DA3MONO-LARGE cho tác vụ độ sâu: dự đoán
  và xác thực zero-shot, không có tuyến huấn luyện.
keywords:
  - Depth Anything 3
  - DA3
  - ước lượng độ sâu đơn ảnh
  - DINOv2
  - độ sâu tương đối
  - depth map
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDepthAnything3l-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Đọc depth map
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDepthAnything3l-depth.pt")

        result = model(SAMPLE_IMAGE)


        depth = result.depth_map    # DepthMap: dense (H, W), cao hơn = gần hơn

        raw = depth.data                # tensor, không có đơn vị mét hoặc tỷ lệ
        xuyên ảnh

        normalized = depth.normalized() # đổi tỷ lệ thành [0, 1] để trực quan
        hóa
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDepthAnything3l-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDepthAnything3l-depth.pt format=onnx

        libreyolo export model=LibreDepthAnything3l-depth.pt format=tensorrt
        half=True
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến dựa trên hậu tố tệp, vì vậy artifact đã xuất tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreDepthAnything3l-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: 0ac96180165c4891
---

## Cài đặt

Depth Anything 3 không cần extra tùy chọn. Mọi thành phần mà mô hình import đều có trong bản cài đặt cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

`result.depth_map` chứa depth map nghịch đảo tương đối dense: giá trị cao hơn nghĩa là gần camera hơn, các giá trị không có đơn vị mét hoặc tỷ lệ xuyên ảnh. Checkpoint upstream phát ra độ sâu tương đối dương; network wrapper của LibreYOLO đảo giá trị này và tái tạo cách xử lý bầu trời chính thức để đầu ra tuân theo hợp đồng độ sâu chung của LibreYOLO. `save=True` ghi bản trực quan hóa áp colormap của map đó ra đĩa; `Results.plot()` không hỗ trợ họ mô hình này vì hàm chỉ được định nghĩa cho pháp tuyến bề mặt và cạnh. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Có một kích thước `l` ở độ phân giải đầu vào cố định. DA3 upstream cũng phát hành các checkpoint any-view Small và Base, một checkpoint độ sâu theo mét, cùng checkpoint Nested và Giant; LibreYOLO không cung cấp checkpoint nào trong số đó. Độ sâu theo mét cần hợp đồng công khai khác với tác vụ độ sâu nghịch đảo tương đối của LibreYOLO, còn checkpoint any-view và Nested cần API camera đa ảnh mà LibreYOLO không cung cấp. Các checkpoint any-view Large và Giant cũng dùng CC-BY-NC-4.0 và không được tham chiếu bởi bất kỳ đường tải xuống nào của LibreYOLO.

Họ mô hình này không cung cấp huấn luyện. `LibreDepthAnything3.train()` luôn phát sinh `NotImplementedError`; hãy huấn luyện ở upstream và chuyển đổi checkpoint DA3MONO-LARGE tương thích bằng `weights/convert_depth_anything3_weights.py`.

## Xác thực

`val()` chạy trình xác thực độ sâu dùng chung: hàm căn chỉnh từng dự đoán với ground truth bằng scale và shift bình phương tối thiểu theo từng ảnh, rồi báo cáo các metric độ sâu tương đối zero-shot tiêu chuẩn là AbsRel, RMSE và ba ngưỡng delta.

<code-tabs name="val" />

## Xuất

<export-matrix />

Việc xuất bị giới hạn ở năm định dạng cho họ mô hình này: ONNX, TorchScript, ExecuTorch, TensorRT và OpenVINO. Yêu cầu bất kỳ định dạng nào khác sẽ phát sinh `NotImplementedError` thay vì thử phép chuyển đổi chưa được xác thực. Artifact đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp `.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng đối tượng `Results`, với `depth_map` thay cho box.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã phát hành cho họ mô hình này.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />

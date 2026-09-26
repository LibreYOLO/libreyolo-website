---
title: Depth Anything V2
families:
  - depth_anything
seo_title: 'Depth Anything V2: dự đoán và xác thực độ sâu đơn ảnh'
description: >-
  Dùng Depth Anything V2 trong LibreYOLO để ước lượng độ sâu đơn ảnh. Cài đặt,
  dự đoán và xác thực; Small dùng Apache-2.0, Base và Large dùng CC-BY-NC-4.0.
lead: >-
  Depth Anything V2 là encoder DINOv2 ghép với decoder DPT để dự đoán depth map
  nghịch đảo tương đối dense từ một ảnh. LibreYOLO hỗ trợ mô hình này cho tác vụ
  độ sâu: dự đoán và xác thực zero-shot, không có tuyến huấn luyện.
keywords:
  - Depth Anything V2
  - ước lượng độ sâu đơn ảnh
  - DPT
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

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDepthAnythingV2s-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Đọc depth map
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")

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

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDepthAnythingV2s-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDepthAnythingV2s-depth.pt format=onnx

        libreyolo export model=LibreDepthAnythingV2s-depth.pt format=tensorrt
        half=True
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến dựa trên hậu tố tệp, vì vậy artifact đã xuất tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreDepthAnythingV2s-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: e1043aba1b70b65c
---

## Cài đặt

Depth Anything V2 không cần extra tùy chọn. Mọi thành phần mà mô hình import đều có trong bản cài đặt cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải về từ Hugging Face trong lần sử dụng đầu tiên và được lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

`result.depth_map` chứa depth map nghịch đảo tương đối dense: giá trị cao hơn nghĩa là gần camera hơn, các giá trị không có đơn vị mét hoặc tỷ lệ xuyên ảnh. `save=True` ghi bản trực quan hóa áp colormap của map đó ra đĩa; `Results.plot()` không hỗ trợ họ mô hình này vì hàm chỉ được định nghĩa cho pháp tuyến bề mặt và cạnh. Độ phân giải đầu vào phải chia hết cho 14, là lưới patch DINOv2 mà DPT head xây dựng trên đó; LibreYOLO kiểm tra điều này trước khi chạy và phát sinh lỗi nếu không đạt. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Có bốn kích thước encoder s/b/l/g, tương ứng với ViT-S/B/L/G. Bảng checkpoint bên dưới chỉ liệt kê s, b và l; không có checkpoint Giant nào được phát hành. Cả bốn dùng chung độ phân giải đầu vào, vì vậy việc chọn kích thước đánh đổi dung lượng encoder chứ không phải kích thước ảnh. Giấy phép cũng là một yếu tố: checkpoint Small dùng Apache-2.0, còn Base và Large dùng CC-BY-NC-4.0, xem phần Giấy phép bên dưới.

Họ mô hình này không cung cấp huấn luyện và tinh chỉnh. `LibreDepthAnythingV2.train()` luôn phát sinh `NotImplementedError`; thay vào đó hãy chuyển đổi một checkpoint upstream tương thích bằng `weights/convert_depth_anything_v2_weights.py`.

## Xác thực

`val()` chạy trình xác thực độ sâu dùng chung: hàm căn chỉnh từng dự đoán với ground truth bằng scale và shift bình phương tối thiểu theo từng ảnh, rồi báo cáo các metric độ sâu tương đối zero-shot tiêu chuẩn là AbsRel, RMSE và ba ngưỡng delta.

<code-tabs name="val" />

## Xuất

<export-matrix />

Artifact đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp `.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng đối tượng `Results`, với `depth_map` thay cho box. Trang [Xuất](/docs/export) liệt kê các đối số mà mọi định dạng chấp nhận.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã phát hành cho họ mô hình này.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />

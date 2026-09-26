---
title: MiDaS
families:
  - midas
seo_title: 'MiDaS: ước lượng độ sâu đơn ảnh trong LibreYOLO'
description: >-
  Dùng MiDaS trong LibreYOLO để ước lượng độ sâu đơn ảnh. Cài đặt, dự đoán, xác
  thực và xuất hai biến thể dùng giấy phép MIT, được tải từ isl-org.
lead: >-
  MiDaS là mô hình ước lượng độ sâu tương đối đơn ảnh được huấn luyện bằng loss
  bất biến theo scale và shift trên các dataset hỗn hợp, hướng nghiên cứu đã
  thiết lập giao thức transfer độ sâu zero-shot mà các họ sau này dùng lại.
  LibreYOLO hỗ trợ mô hình cho tác vụ độ sâu: dự đoán và xác thực zero-shot,
  không có tuyến huấn luyện.
keywords:
  - MiDaS
  - ước lượng độ sâu đơn ảnh
  - DPT
  - độ sâu tương đối
  - depth map
  - độ sâu zero-shot
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Khi chưa có trên đĩa: LibreYOLO tải từ bản phát hành GitHub chính thức
        # của isl-org/MiDaS và kiểm tra theo SHA-256 cố định trước khi sử dụng.
        model = LibreYOLO("LibreMiDaSl-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMiDaSl-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Biến thể Small
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Encoder EfficientNet-Lite3, nhỏ và nhanh hơn kích thước l DPT-Large.
        model = LibreYOLO("LibreMiDaSs-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMiDaSl-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMiDaSl-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMiDaSl-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMiDaSl-depth.pt format=onnx
        libreyolo export model=LibreMiDaSl-depth.pt format=tensorrt half=True
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến dựa trên hậu tố tệp, vì vậy artifact đã xuất tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreMiDaSl-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: ce2fbf3ae43e9be4
---

## Cài đặt

MiDaS không cần extra tùy chọn. Mọi thành phần mà mô hình import đều có trong bản cài đặt cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

MiDaS là họ độ sâu duy nhất mà LibreYOLO không phát hành lại trên tổ chức Hugging Face riêng. Yêu cầu checkpoint bằng tên tệp LibreYOLO sẽ tải trực tiếp artifact chính thức tương ứng từ các bản phát hành GitHub `isl-org/MiDaS`, kiểm tra theo SHA-256 cố định và bọc bằng metadata checkpoint của LibreYOLO trước lần sử dụng đầu tiên; các lượt sau dùng lại tệp cục bộ trong bộ nhớ đệm. Xem phần Giấy phép để biết lý do.

<code-tabs name="predict" />

`result.depth_map` chứa depth map nghịch đảo tương đối dense: giá trị cao hơn nghĩa là gần camera hơn, các giá trị không có đơn vị mét hoặc tỷ lệ xuyên ảnh. `save=True` ghi bản trực quan hóa áp colormap của map đó ra đĩa; `Results.plot()` không hỗ trợ họ mô hình này vì hàm chỉ được định nghĩa cho pháp tuyến bề mặt và cạnh. Xem [dự đoán](/docs/predict) để biết về nguồn, xử lý luồng và kết quả.

## Biến thể

Có hai biến thể với encoder khác nhau, không chỉ là các scale khác nhau của cùng một encoder. `s` là MiDaS v2.1 Small, một encoder EfficientNet-Lite3. `l` là DPT-Large, encoder ViT-L/16 với decoder DPT mà MiDaS đưa vào cho dự đoán dense. Chúng cũng tiền xử lý khác nhau: `s` dùng phép đổi kích thước theo tỷ lệ với giới hạn trên cùng chuẩn hóa mean/std ImageNet, còn `l` dùng phép đổi kích thước theo tỷ lệ tối thiểu với mean và std 0.5. Chọn `s` cho CNN nhẹ hơn, `l` cho độ chính xác của transformer decoder.

Họ mô hình này không cung cấp huấn luyện. `LibreMiDaS.train()` luôn phát sinh `NotImplementedError`.

## Xác thực

`val()` chạy trình xác thực độ sâu dùng chung: hàm căn chỉnh từng dự đoán với ground truth bằng scale và shift bình phương tối thiểu theo từng ảnh, rồi báo cáo các metric độ sâu tương đối zero-shot tiêu chuẩn là AbsRel, RMSE và ba ngưỡng delta.

<code-tabs name="val" />

## Xuất

<export-matrix />

Artifact đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp `.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng đối tượng `Results`, với `depth_map` thay cho box.

<code-tabs name="export" />

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />



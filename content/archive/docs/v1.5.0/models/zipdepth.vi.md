---
title: ZipDepth
families:
  - zipdepth
seo_title: 'ZipDepth: độ sâu đơn mắt nhẹ trong LibreYOLO'
description: >-
  Dùng ZipDepth trong LibreYOLO để ước lượng độ sâu đơn mắt nhẹ. Cài đặt, dự
  đoán, đánh giá và xuất hai checkpoint theo giấy phép MIT.
lead: >-
  ZipDepth là CNN nhỏ gọn có thể tái tham số hóa, được chưng cất từ Depth
  Anything V2 Large để dự đoán bản đồ nghịch đảo độ sâu tương đối dày đặc.
  LibreYOLO hỗ trợ mô hình cho tác vụ độ sâu: dự đoán và đánh giá zero-shot,
  không có luồng huấn luyện.
keywords:
  - ZipDepth
  - ước lượng độ sâu đơn mắt
  - mô hình độ sâu cho thiết bị biên
  - độ sâu tương đối
  - bản đồ độ sâu
  - CNN tái tham số hóa
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreZipDepthb-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Checkpoint NPU/thiết bị biên
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Cùng bộ mã hóa, với head upsampling không dùng unfold cho compiler
        thiếu

        # hỗ trợ gather/unfold. Đầu ra tương đương về mặt thị giác với
        checkpoint b.

        model = LibreYOLO("LibreZipDepthbnpu-depth.pt")

        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreZipDepthb-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        model.export(format="onnx")
        model.export(format="ncnn")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreZipDepthb-depth.pt format=onnx
        libreyolo export model=LibreZipDepthbnpu-depth.pt format=ncnn
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến theo hậu tố tệp, nên artifact đã xuất được tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreZipDepthb-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: 891eaa1a42795a4c
---

## Cài đặt

ZipDepth không cần extra tùy chọn. Mọi thành phần được import đều có trong bản
cài đặt cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải từ Hugging Face ở lần dùng đầu tiên và lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

`result.depth_map` chứa bản đồ nghịch đảo độ sâu tương đối dày đặc: giá trị cao
hơn nghĩa là gần camera hơn, và các giá trị không có đơn vị mét hay tỉ lệ chung
giữa các ảnh. `save=True` ghi ảnh trực quan hóa theo bản đồ màu của bản đồ này
ra đĩa; `Results.plot()` không bao quát họ này vì chỉ được định nghĩa cho pháp
tuyến bề mặt và cạnh. Xem [dự đoán](/docs/predict) để biết về nguồn, streaming
và xử lý kết quả.

## Các biến thể

Có hai checkpoint với cùng dung lượng bộ mã hóa, chỉ khác head upsampling đã
huấn luyện. `b` dùng upsampling lồi và chạy trên GPU hoặc CPU. `bnpu` thay bằng
decoder không dùng unfold cho NPU và compiler thiết bị biên thiếu hỗ trợ
gather/unfold; đầu ra được ghi nhận là tương đương về mặt thị giác với `b`.
Chọn `bnpu` khi đích xuất là runtime bị giới hạn, còn lại chọn `b`.

Cả hai checkpoint đều được chưng cất từ nhãn giả của Depth Anything V2 Large,
vì vậy họ này là tầng nhỏ gọn hướng đến thiết bị biên của tác vụ độ sâu trong
LibreYOLO, bên cạnh các bộ mã hóa Depth Anything V2 lớn hơn.

Họ này không cung cấp huấn luyện. `LibreZipDepth.train()` luôn phát sinh
`NotImplementedError`: công thức thượng nguồn chưng cất nhãn giả trên một tập
ảnh lớn, không thể tái tạo dưới dạng một lượt huấn luyện LibreYOLO. Hãy huấn luyện
ở thượng nguồn tại [fabiotosi92/ZipDepth](https://github.com/fabiotosi92/ZipDepth)
và chuyển đổi kết quả bằng `weights/convert_zipdepth_weights.py`.

## Đánh giá

`val()` chạy bộ đánh giá độ sâu dùng chung: phương thức căn chỉnh từng dự đoán
với ground truth bằng hệ số tỉ lệ và độ dịch chuyển bình phương tối thiểu theo
từng ảnh, sau đó báo cáo các chỉ số độ sâu tương đối zero-shot tiêu chuẩn gồm
AbsRel, RMSE và ba ngưỡng delta.

<code-tabs name="val" />

## Xuất

<export-matrix />

Xuất tuân theo giao diện dày đặc có độ phân giải cố định: ảnh nguồn được co giãn
đến canvas đã xuất và bản đồ độ sâu trả về được đổi lại về canvas gốc sau đó.
Artifact đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp
`.onnx` hoặc `.ncnn` hoạt động như checkpoint và trả về cùng `Results`, với
`depth_map` thay cho các box.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã công bố cho họ này.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />


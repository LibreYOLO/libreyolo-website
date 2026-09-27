---
title: PicoSAM3
families:
  - picosam3
seo_title: 'PicoSAM3: phân đoạn biên bằng prompt box trong LibreYOLO'
description: >-
  Dùng PicoSAM3 trong LibreYOLO để phân đoạn vùng bằng prompt box trên cảm biến
  biên. Cài đặt, dự đoán và xuất checkpoint pico theo Apache-2.0.
lead: >-
  PicoSAM3 là CNN nhỏ gọn được chưng cất từ SAM 2.1 và SAM 3, xây dựng để phân
  đoạn vùng quan tâm bằng prompt box trên các cảm biến như Sony IMX500.
  LibreYOLO hỗ trợ mô hình qua factory LibreSAM chuyên dụng, tách biệt với
  factory detector LibreYOLO(), và chỉ dùng prompt box.
keywords:
  - PicoSAM3
  - Segment Anything
  - phân đoạn biên
  - vùng quan tâm
  - prompt box
  - suy luận trên cảm biến
  - IMX500
  - chưng cất tri thức
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompt box
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # PicoSAM3 chỉ có một kích thước "pico", nên không cần bí danh khác.

        model = LibreSAM("picosam3")


        # bboxes= là prompt duy nhất được hỗ trợ: [x1, y1, x2, y2] hoặc danh
        sách

        # các box, mỗi box cho một mask. Mỗi box được nới rộng 10%, đưa về hình

        # vuông, cắt theo ảnh và đổi kích thước thành 96x96 trước khi CNN chạy.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])

        print(result.masks.xy)      # đa giác cho mỗi mask

        print(result.boxes.xyxy)    # box khít suy ra từ mask
    - label: 'Mã hóa một lần, dùng nhiều prompt'
      language: python
      code: >
        from libreyolo import LibrePicoSAM3, SAMPLE_IMAGE


        model = LibrePicoSAM3()


        # set_image() lưu ảnh nguồn vào bộ nhớ đệm; PicoSAM3 chạy một forward

        # đầy đủ của CNN cho mỗi box, nên thao tác này tiết kiệm việc tải/giải
        mã

        # ảnh chứ không phải lượt chạy bộ mã hóa như với các họ SAM khác.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(bboxes=[300, 200, 900, 700])

        b = model.predict(bboxes=[100, 100, 400, 400])

        model.reset_image()
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibrePicoSAM3

        model = LibrePicoSAM3()
        model.export(format="onnx", output_path="LibrePicoSAM3pico.onnx")

        # opset (mặc định 13) và dynamic (mặc định True, chỉ trục batch) là
        # các đối số xuất duy nhất họ này chấp nhận.
    - label: Dùng tệp đã xuất
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # PicoSAM3 xuất CNN ROI 96x96 thô: roi_image -> mask_logits.

        # Không có bước tiền/hậu xử lý phía LibreYOLO để dùng lại ở đây vì

        # export() không được định tuyến trở lại qua LibreYOLO() như một

        # checkpoint detector.

        session = ort.InferenceSession("LibrePicoSAM3pico.onnx")

        name = session.get_inputs()[0].name

        outputs = session.run(None, {name: np.zeros((1, 3, 96, 96),
        dtype=np.float32)})


        for meta, array in zip(session.get_outputs(), outputs):
            print(meta.name, array.shape)
source_hash: 5d60ff14fe61ba29
---

## Cài đặt

PicoSAM3 cần extra `sam`: thao tác tải trọng số của LibreYOLO vẫn đi qua công
cụ Hugging Face của `transformers`, dù suy luận chạy trên CNN native không dùng
`transformers`.

```bash
pip install "libreyolo[sam]"
```

## Dự đoán

`LibreSAM(...)` (hoặc `LibrePicoSAM3(...)` dành riêng cho họ) là một điểm vào
riêng, khác với `LibreYOLO(...)`: nó trả về mô hình phân đoạn dùng prompt thay
vì detector, vì forward pass ở đây không có ý nghĩa nếu thiếu prompt. Không có
lệnh CLI `libreyolo predict` cho họ này; hãy dùng Python API.

<code-tabs name="predict" />

PicoSAM3 chỉ chấp nhận `bboxes=`; truyền `points=`, `labels=`, `masks=`, `text=`,
`multimask=True` hoặc bỏ box để phân đoạn mọi thứ đều phát sinh `ValueError` rõ
ràng vì các chế độ đó không tồn tại trong mô hình thượng nguồn. `conf` lọc theo
chất lượng mask dự đoán (IoU), không phải độ tin cậy phát hiện, và phải nằm trong
khoảng `0.0` đến `1.0`. Mọi mask mang class id `0`, tên `"object"`. `train()`,
`val()` và `track()` phát sinh `NotImplementedError`; hãy dùng LibreSAM2 hoặc
LibreSAM3 cho prompt điểm, văn bản, mask hoặc phân đoạn mọi thứ. Xem
[dự đoán](/docs/predict) để biết các loại nguồn.

## Các biến thể

Có một kích thước là pico với đầu vào ROI cố định 96 px: PicoSAM3 chạy một
forward CNN đầy đủ cho mỗi box thay vì mã hóa toàn bộ ảnh một lần.

## Xuất

<export-matrix />

PicoSAM3 là họ duy nhất trong tầng SAM hỗ trợ xuất: mô hình đưa CNN ROI 96x96 thô
sang ONNX, `roi_image -> mask_logits`, không nhúng NMS hay hậu xử lý mask. Các họ
SAM khác phát sinh `NotImplementedError` trên `export()` vì phần tách
encoder/decoder của chúng chưa có giao diện xuất runtime được định nghĩa. Đồ thị
PicoSAM3 đã xuất không tải lại qua `LibreYOLO()`; hãy chạy trực tiếp bằng runtime
như `onnxruntime`, áp dụng cùng bước tiền xử lý ROI vuông có padding 10% nêu trên.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã công bố cho họ này.

<checkpoint-table />

## Giấy phép

<provenance-box>

PicoSAM3 được chưng cất từ SAM 2.1 và SAM 3 đóng vai trò mô hình giáo viên.
LibreYOLO không đóng gói hay phân phối lại mã hoặc trọng số của hai mô hình giáo
viên trong họ này; chỉ CNN học sinh nhỏ gọn và checkpoint đã chuyển đổi được
phân phối.

</provenance-box>

## Trích dẫn

<citation-block />


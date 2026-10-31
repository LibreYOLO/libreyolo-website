---
title: PP-OCRv5
families:
  - ppocr
seo_title: 'PP-OCRv5: phát hiện và nhận dạng văn bản trong LibreYOLO'
description: >-
  Dùng PP-OCRv5 trong LibreYOLO để OCR văn bản cảnh đa ngôn ngữ. Cài đặt, dự
  đoán và đánh giá các checkpoint t và l theo giấy phép Apache-2.0.
lead: >-
  PP-OCRv5 là pipeline phát hiện và nhận dạng văn bản của PaddleOCR: detector
  nhị phân hóa khả vi định vị các tứ giác văn bản và bộ nhận dạng SVTR/CTC đọc
  chúng. LibreYOLO chuyển mô hình sang PyTorch với hai tầng.
keywords:
  - PP-OCRv5
  - PaddleOCR
  - OCR
  - phát hiện văn bản
  - nhận dạng văn bản
  - chữ trong cảnh
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for text, conf in zip(result.ocr.texts, result.ocr.conf):
            print(text, float(conf))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibrePPOCRl-ocr.pt source=receipt.jpg save=True
    - label: Tứ giác
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        result = model(SAMPLE_IMAGE)

        # Đa giác (N, 4, 2) theo thứ tự đọc: trên-trái, trên-phải,
        # dưới-phải, dưới-trái. Tứ giác phát hiện là đa giác thực
        # (văn bản xoay), nên nằm trong result.ocr chứ không phải result.boxes.
        print(result.ocr.data.shape)
        print(result.ocr.det_conf)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        metrics = model.val(data="my-dataset")

        print(metrics["metrics/det_hmean"])
        print(metrics["metrics/e2e_f1"])       # chỉ số chính
        print(metrics["metrics/rec_1-NED"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePPOCRl-ocr.pt data=my-dataset
source_hash: 9835057f8bd95bc1
---

## Cài đặt

PP-OCRv5 không cần extra ngoài gói cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải từ Hugging Face ở lần dùng đầu tiên và lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Mỗi checkpoint gói cả hai giai đoạn phát hiện và nhận dạng trong một tệp `.pt`,
với bộ ký tự nhận dạng và giá trị mặc định của pipeline nằm trong metadata
checkpoint. Bộ nhận dạng đọc tiếng Trung giản thể và phồn thể, tiếng Anh, tiếng
Nhật và pinyin bằng một từ điển. `result.ocr` là payload `OCRRegions`: `.data`
chứa các đa giác bốn điểm, `.texts` chứa văn bản nhận dạng, `.conf` chứa điểm nhận
dạng theo vùng và `.det_conf` chứa điểm phát hiện. Nguồn nhiều ảnh chạy tuần tự:
pipeline hai giai đoạn không tạo batch giữa các ảnh. Xem
[dự đoán](/docs/predict) để biết về nguồn, streaming và xử lý kết quả.

## Các biến thể

Có hai tầng: `t`, xây dựng trên các backbone PP-LCNetV3/PP-OCRv5_mobile nhẹ hơn
để dùng với CPU, và `l`, xây dựng trên backbone server PP-HGNetV2 để có độ chính
xác cao hơn. Cả hai tầng chạy phát hiện với giới hạn cạnh dài cố định và nhận dạng
các crop theo batch; `rec_batch` điều khiển số crop đi qua bộ nhận dạng trong mỗi
forward pass.

## Đánh giá

`val()` đo pipeline trên một thư mục ảnh cùng tệp `labels/<split>.jsonl` hoặc YAML
tập dữ liệu tương đương; mỗi nhãn liệt kê các đa giác vùng văn bản và nội dung
của chúng theo từng ảnh. Phương thức báo cáo hmean phát hiện (precision/recall/F1
được ghép theo IoU), F1 đầu cuối (hmean cộng với kết quả khớp chính xác văn bản
sau chuẩn hóa, là chỉ số fitness của checkpoint) và 1-NED, tức khoảng cách chỉnh
sửa chuẩn hóa trung bình trên các cặp đã ghép.

<code-tabs name="val" />

## Xuất

<export-matrix />

PP-OCRv5 là pipeline hai mạng, trong đó phát hiện và nhận dạng di chuyển cùng
nhau, không phải một đồ thị có thể truy vết; chức năng xuất chưa được triển khai:
chưa hỗ trợ định dạng nào. Hãy tinh chỉnh trực tiếp mã huấn luyện thượng nguồn
theo Apache-2.0 và chuyển đổi kết quả bằng `weights/convert_ppocr_weights.py`
nếu bạn cần checkpoint ngoài định dạng này.

## Checkpoint

Mọi tệp trọng số đã công bố cho họ này.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />


---
title: OCR
seo_title: 'OCR: phát hiện và nhận dạng văn bản trong LibreYOLO'
description: >-
  Tìm và đọc văn bản trong ảnh bằng LibreYOLO. Dự đoán tứ giác và transcript,
  gán nhãn dataset JSONL, rồi xác thực bằng hmean, F1 đầu cuối và 1-NED.
lead: >-
  OCR định vị văn bản trong ảnh và đọc nội dung. LibreYOLO cung cấp dưới dạng
  tác vụ ocr, trả về một polygon bốn điểm cùng một transcript cho mỗi vùng văn
  bản, theo thứ tự đọc.
keywords:
  - thư viện ocr python
  - nhận dạng chữ trong ảnh
  - phát hiện text tứ giác
  - PP-OCRv5 python
  - nhận dạng văn bản đầu cuối
last_verified: 1.5.0
snippets:
  predict:
    - label: Đọc văn bản trong ảnh
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Tầng t nhẹ hơn trong hai tầng, được thiết kế cho CPU. SAMPLE_IMAGE
        # giúp ví dụ chạy được; hãy trỏ tới ảnh có văn bản của riêng bạn.
        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        result = model(SAMPLE_IMAGE)

        regions = result.ocr
        print(len(regions), "regions")
        for text, score in zip(regions.texts, regions.conf):
            print(repr(text), float(score))
    - label: Đọc các tứ giác
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        result = model(SAMPLE_IMAGE)

        regions = result.ocr
        print(regions.data.shape)   # polygon (N, 4, 2), TL TR BR BL
        print(regions.xyxy)         # bao lồi thẳng trục của các polygon đó
        print(regions.det_conf)     # điểm phát hiện, tách biệt với .conf
    - label: Lọc theo độ tin cậy nhận dạng
      language: python
      code: |
        import numpy as np
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        result = model(SAMPLE_IMAGE)

        # Lập index bằng vị trí, không phải mask boolean: slicing mang theo
        # transcript và cả hai mảng điểm số cùng phần hình học.
        regions = result.ocr.numpy()
        keep = regions[np.flatnonzero(regions.conf >= 0.9)]
        print(keep.texts)
  val:
    - label: Xác thực và đọc các key metric
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        metrics = model.val(data="my-ocr-dataset")

        print(metrics["metrics/det_precision"], metrics["metrics/det_recall"])
        print(metrics["metrics/det_hmean"])
        print(metrics["metrics/e2e_f1"])       # fitness
        print(metrics["metrics/rec_1-NED"])
source_hash: 58ad5305c9dd458c
---

## Định nghĩa

Tác vụ `ocr` thực hiện hai việc trong một lời gọi: định vị mọi vùng văn bản trong
ảnh và chuyển chúng thành ký tự. Vùng được trả về dưới dạng polygon bốn điểm
thay vì hộp thẳng trục vì văn bản trong cảnh thường bị xoay, và theo thứ tự đọc,
từ trên xuống dưới rồi từ trái sang phải.

Dự đoán điền `result.ocr`, một payload `OCRRegions`. `.data` là mảng float
`(N, 4, 2)` gồm các polygon theo pixel ảnh gốc, được sắp theo trên-trái,
trên-phải, dưới-phải, dưới-trái; `.texts` là danh sách N transcript; `.conf` là
điểm nhận dạng theo vùng và `.det_conf` là điểm phát hiện; `.xyxy` cho bao lồi
thẳng trục của mỗi polygon. Vì các tứ giác là polygon thực, chúng không điền
`result.boxes`. Thao tác slicing `OCRRegions` mang transcript và cả hai mảng
điểm số đi cùng phần hình học.

## Mô hình

Hai family phục vụ `ocr`.

[PP-OCRv5](/docs/models/pp-ocrv5) là pipeline chuyên dụng: detector
differentiable binarization tìm tứ giác văn bản và recognizer SVTR/CTC đọc chúng,
với cả hai giai đoạn được gói trong một tệp `.pt` cùng bộ ký tự nhận dạng. Mô
hình có hai tầng, một tầng nhẹ hơn cho CPU và một tầng server có độ chính xác
cao hơn, còn một dictionary bao phủ tiếng Trung giản thể và phồn thể, tiếng Anh,
tiếng Nhật và pinyin.

[SenseNova-Vision](/docs/models/sensenova-vision) thực hiện OCR bằng cách sinh từ
dưới dạng văn bản có tag từ cùng checkpoint 7B phục vụ sáu tác vụ khác, được nạp
bằng `LibreVLM("sensenova-vision", task="ocr")`. Nó cần thành phần bổ sung
`sensenova`, còn trọng số bị giới hạn cho mục đích phi thương mại; giấy phép nằm
trên trang của mô hình.

## Dự đoán

Trọng số được tải từ Hugging Face trong lần sử dụng đầu tiên và lưu vào cache
cục bộ.

<code-tabs name="predict" />

PP-OCRv5 chạy phát hiện ở giới hạn cạnh dài cố định rồi nhận dạng các vùng crop
theo batch, với `rec_batch` điều khiển số vùng crop đi qua recognizer trên mỗi
forward pass. Nguồn nhiều ảnh chạy tuần tự vì pipeline hai giai đoạn không tạo
batch xuyên ảnh. Xem [dự đoán](/docs/predict) để biết về nguồn, stream và cách
xử lý kết quả.

## Định dạng dataset

Nhãn OCR là một tệp JSONL cho mỗi split, mỗi ảnh một đối tượng JSON, nằm bên cạnh
chính các ảnh.

```text
my-ocr-dataset/
  images/
    val/receipt.jpg
  labels/
    val.jsonl
```

Mỗi dòng đặt tên ảnh và liệt kê các vùng:

```json
{"image": "receipt.jpg", "regions": [{"polygon": [[10, 12], [118, 14], [117, 40], [9, 38]], "text": "TOTAL 12.50"}]}
```

`polygon` là tứ giác bốn điểm theo tọa độ pixel tuyệt đối, được sắp theo
trên-trái, trên-phải, dưới-phải, dưới-trái. Vùng có văn bản không đọc được được
gán nhãn `"text": "###"`, quy ước không quan tâm của ICDAR: vùng bị loại khỏi
điểm nhận dạng, còn dự đoán chồng lên nó được bỏ qua thay vì tính là false
positive.

Chỉ cần truyền thư mục gốc làm `data=`. YAML dataset là phương án thay thế, gồm
`path` cùng tên thư mục `images` và `labels` tùy chọn, cùng `nc: 1` và
`names: {0: text}` làm placeholder schema vì mô hình OCR trả về `Results.ocr`
thay vì kết quả phát hiện. Xem [định dạng
dataset](/docs/reference/dataset-formats) để biết hợp đồng đầy đủ.

## Huấn luyện

Không family OCR nào có implementation huấn luyện: `train()` phát sinh
`NotImplementedError` trên cả hai, còn hỗ trợ OCR chỉ bao gồm dự đoán và xác
thực. Trang PP-OCRv5 nêu tên mã huấn luyện upstream dùng Apache-2.0 và script
chuyển đổi đưa checkpoint đã tinh chỉnh trở lại LibreYOLO.

## Xác thực

`val()` tính điểm toàn bộ pipeline, cả phát hiện và nhận dạng, bằng cách khớp
polygon dự đoán với polygon ground truth theo kiểu một-một ở IoU trên 0.5.

<code-tabs name="val" />

`metrics/det_precision`, `metrics/det_recall` và `metrics/det_hmean` chỉ tính
điểm định vị: một cặp khớp chỉ cần polygon chồng lấn, bất kể transcript ghi gì.
`metrics/e2e_precision`, `metrics/e2e_recall` và `metrics/e2e_f1` bổ sung phần
đọc: một cặp khớp cần cùng độ chồng lấn polygon và transcript khớp chính xác sau
khi chuẩn hóa NFKC và bỏ khoảng trắng, còn phép so sánh vẫn phân biệt chữ hoa
chữ thường. `metrics/e2e_f1` cũng là `fitness`, con số mà cơ chế chọn checkpoint
tốt nhất đọc.

`metrics/rec_1-NED` chấm điểm riêng recognizer trên các cặp mà phát hiện đã
khớp: một trừ normalized edit distance, vì vậy transcript sai một ký tự có điểm
gần 1 trong khi F1 đầu cuối cho điểm 0.

## Xuất

Tác vụ này không có định dạng xuất. PP-OCRv5 là hai mạng di chuyển cùng nhau
thay vì một graph có thể trace, và `export()` phát sinh lỗi cho mọi định dạng
trên cả hai family. Để triển khai bên ngoài LibreYOLO, hãy tinh chỉnh ở upstream
và dùng đường dẫn triển khai upstream.

---
title: OWLv2
families:
  - owlv2
seo_title: 'OWLv2 trong LibreYOLO: phát hiện đối tượng zero-shot'
description: >-
  Dùng OWLv2 trong LibreYOLO để phát hiện mọi đối tượng được mô tả bằng văn bản.
  Cài đặt extra openvocab và dự đoán bằng từ vựng văn bản tự do.
lead: >-
  OWLv2 là detector đối tượng với từ vựng mở do Google Research phát triển, chấm
  điểm các vùng ảnh theo embedding văn bản từ encoder kiểu CLIP. LibreYOLO bọc
  mô hình thành họ chỉ dành cho dự đoán trong cấp detector với từ vựng mở.
keywords:
  - OWLv2
  - OWL-ViT
  - phát hiện đối tượng với từ vựng mở
  - phát hiện zero-shot
  - detector điều kiện hóa bằng văn bản
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("owlv2-b16")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.1)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Từ vựng mặc định
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        # Bỏ qua set_classes() sẽ giữ từ vựng COCO-80 mặc định của cấp.
        model = LibreOpenVocab("owlv2-l14")
        result = model.predict(SAMPLE_IMAGE, conf=0.1)
        print(result.names)
source_hash: 2d0ce68af0daabb7
---

## Cài đặt

OWLv2 tải qua cấp detector với từ vựng mở của LibreYOLO, cấp này cần extra `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

Extra đó kéo về `transformers` và `timm`, các thư viện Hugging Face mà cấp này sử dụng.

## Dự đoán

OWLv2 không phải checkpoint mà LibreYOLO tải qua `LibreYOLO()`. Mô hình tải qua factory `LibreOpenVocab` cùng cấp, factory này tải snapshot Hugging Face trong lần sử dụng đầu tiên và lưu vào bộ nhớ đệm dưới `weights/`.

<code-tabs name="predict" />

`set_classes()` đặt từ vựng văn bản được giữ lại: gọi lại để thay danh sách hoặc bỏ qua để giữ nhãn COCO-80 mặc định. Mỗi nhãn được bọc trong mẫu prompt cố định trước khi đến text tower, phù hợp với cách `Owlv2ForObjectDetection` của `transformers` được huấn luyện.

OWLv2 không có ngưỡng text-token: chỉ `conf` lọc detection, còn truyền `text_threshold` sẽ phát sinh lỗi. `iou` được chấp nhận để tương thích API nhưng phát cảnh báo và không làm gì vì không có non-maximum suppression ở đây. `imgsz` và `augment=True` bị từ chối hoàn toàn: processor của `transformers` sở hữu bước đổi kích thước, còn tăng cường khi kiểm thử nằm ngoài phạm vi của cấp này. `predict()` trên một ảnh trả về một `Results`, không phải danh sách; truyền thư mục, danh sách ảnh hoặc `stream=True` cho nguồn video để nhận nhiều kết quả. Không có tuyến CLI cho họ mô hình này; `libreyolo predict` chỉ tải checkpoint `.pt` qua `LibreYOLO()`, vì vậy các họ `LibreOpenVocab` chạy từ Python. Xem [dự đoán](/docs/predict) để biết loại nguồn và xử lý luồng.

## Biến thể

Có hai checkpoint `b16` (base, kích thước patch 16) và `l14` (large, kích thước patch 14). `b16` là kích thước mặc định của cấp này khi không chỉ định. Cả hai mirror bản phát hành Google Research chính thức qua `Owlv2ForObjectDetection` của `transformers`, được tải một lần vào snapshot Hugging Face do LibreYOLO host và giữ nguyên các tệp upstream. Chưa có số liệu độ chính xác hoặc độ trễ nào được công bố cho họ mô hình này.

Huấn luyện, xác thực tập dữ liệu (dataset) và xuất đều nằm ngoài phạm vi của cấp này: `train()`, `val()` và `export()` luôn phát sinh `NotImplementedError`. Đây là wrapper chỉ dành cho dự đoán quanh checkpoint đã phát hành.

## Checkpoint

Mọi tệp trọng số đã phát hành cho họ mô hình này.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />



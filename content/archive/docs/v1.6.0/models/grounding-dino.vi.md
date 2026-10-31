---
title: Grounding DINO
families:
  - grounding_dino
seo_title: 'Grounding DINO trong LibreYOLO: phát hiện open-set'
description: >-
  Dùng Grounding DINO trong LibreYOLO để phát hiện mọi đối tượng được mô tả bằng
  văn bản. Cài đặt extra openvocab và dự đoán với từ vựng văn bản tự do.
lead: >-
  Grounding DINO là detector đối tượng open-set do IDEA Research phát triển,
  chấm điểm ảnh theo prompt văn bản tự do thay vì danh sách lớp đối tượng cố
  định. LibreYOLO bọc mô hình thành họ chỉ dành cho dự đoán trong cấp detector
  với từ vựng mở.
keywords:
  - Grounding DINO
  - phát hiện đối tượng với từ vựng mở
  - phát hiện open-set
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

        model = LibreOpenVocab("grounding-dino-t")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Ngưỡng văn bản
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-b")
        model.set_classes(["remote control", "school bus"])

        # conf lọc theo điểm box, text_threshold lọc theo điểm token của cụm từ
        # đã giải mã. Cả hai mặc định là 0.25 khi không được đặt.
        result = model.predict(SAMPLE_IMAGE, conf=0.25, text_threshold=0.3)
        print(result.names)
source_hash: 06bd13b8e6a66038
---

## Cài đặt

Grounding DINO tải qua cấp detector với từ vựng mở của LibreYOLO, cấp này cần extra `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

Extra đó kéo về `transformers` và `timm`, các thư viện Hugging Face mà cấp này sử dụng.

## Dự đoán

Grounding DINO không phải checkpoint mà LibreYOLO tải qua `LibreYOLO()`. Mô hình tải qua factory `LibreOpenVocab` cùng cấp, factory này tải snapshot Hugging Face trong lần sử dụng đầu tiên và lưu vào bộ nhớ đệm dưới `weights/`.

<code-tabs name="predict" />

`set_classes()` đặt từ vựng văn bản được giữ lại: gọi lại để thay danh sách hoặc bỏ qua để giữ nhãn COCO-80 mặc định. Grounding DINO giải mã các cụm từ tự do từ đầu ra văn bản của chính nó rồi tự ánh xạ về từ vựng đó; kết quả khớp chuẩn hóa chính xác được ưu tiên, kết quả khớp toàn token được chấp nhận, còn cụm từ mơ hồ hoặc không khớp sẽ bị bỏ thay vì đoán, vì vậy `school bus` không bao giờ được ánh xạ riêng thành `bus` hoặc `school`. Từ vựng đủ dài để vượt giới hạn token của text encoder được chia thành nhiều prompt, chạy như các forward pass riêng và hợp nhất lại thành một tập detection bị giới hạn bởi `max_det`.

`iou` được chấp nhận để tương thích API nhưng phát cảnh báo và không làm gì vì không có non-maximum suppression ở đây. `imgsz` và `augment=True` bị từ chối hoàn toàn: processor của `transformers` sở hữu bước đổi kích thước, còn tăng cường khi kiểm thử nằm ngoài phạm vi của cấp này. `predict()` trên một ảnh trả về một `Results`, không phải danh sách; truyền thư mục, danh sách ảnh hoặc `stream=True` cho nguồn video để nhận nhiều kết quả. Không có tuyến CLI cho họ mô hình này; `libreyolo predict` chỉ tải checkpoint `.pt` qua `LibreYOLO()`, vì vậy các họ `LibreOpenVocab` chạy từ Python. Xem [dự đoán](/docs/predict) để biết loại nguồn và xử lý luồng.

## Biến thể

Có hai checkpoint `t` và `b`. `t` là kích thước mặc định của cấp này khi không chỉ định. Cả hai mirror bản phát hành chính thức của IDEA Research qua `GroundingDinoForObjectDetection` trong `transformers`, được tải một lần vào snapshot Hugging Face do LibreYOLO host và giữ nguyên các tệp upstream. Chưa có số liệu độ chính xác hoặc độ trễ nào được công bố cho họ mô hình này.

Huấn luyện, xác thực tập dữ liệu (dataset) và xuất đều nằm ngoài phạm vi của cấp này: `train()`, `val()` và `export()` luôn phát sinh `NotImplementedError`. Đây là wrapper chỉ dành cho dự đoán quanh checkpoint đã phát hành.

## Checkpoint

Mọi tệp trọng số đã phát hành cho họ mô hình này.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />



---
title: OMDet-Turbo
families:
  - omdet_turbo
seo_title: 'OMDet-Turbo trong LibreYOLO: phát hiện zero-shot thời gian thực'
description: >-
  Dùng OMDet-Turbo trong LibreYOLO để phát hiện với từ vựng mở theo thời gian
  thực. Cài đặt extra openvocab và dự đoán bằng từ vựng văn bản tự do.
lead: >-
  OMDet-Turbo là detector đối tượng với từ vựng mở theo thời gian thực do Om AI
  Lab phát triển, tách embedding lớp đối tượng khỏi prompt tác vụ ngôn ngữ.
  LibreYOLO bọc mô hình thành họ chỉ dành cho dự đoán trong cấp detector với từ
  vựng mở.
keywords:
  - OMDet-Turbo
  - OmDet
  - phát hiện đối tượng với từ vựng mở
  - phát hiện thời gian thực
  - phát hiện zero-shot
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("omdet-turbo")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.3)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Ngưỡng NMS tùy chỉnh
      language: python
      code: >
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE


        model = LibreOpenVocab("omdet-turbo")

        model.set_classes(["traffic light", "bicycle"])


        # OMDet-Turbo là họ duy nhất trong cấp này tôn trọng iou=: bước hậu xử
        lý

        # riêng nhận ngưỡng loại bỏ làm đối số, mặc định là 0.5 khi không đặt
        iou=.

        result = model.predict(SAMPLE_IMAGE, conf=0.3, iou=0.7)

        print(result.names, len(result))
source_hash: c2a375d234341b7e
---

## Cài đặt

OMDet-Turbo tải qua cấp detector với từ vựng mở của LibreYOLO, cấp này cần extra `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

Extra đó kéo về `transformers` và `timm`, các thư viện Hugging Face mà cấp này sử dụng; backbone Swin của OMDet-Turbo tải qua wrapper `TimmBackbone` của `transformers`.

## Dự đoán

OMDet-Turbo không phải checkpoint mà LibreYOLO tải qua `LibreYOLO()`. Mô hình tải qua factory `LibreOpenVocab` cùng cấp, factory này tải snapshot Hugging Face trong lần sử dụng đầu tiên và lưu vào bộ nhớ đệm dưới `weights/`.

<code-tabs name="predict" />

`set_classes()` đặt từ vựng văn bản được giữ lại: gọi lại để thay hoàn toàn danh sách hoặc bỏ qua để giữ nhãn COCO-80 mặc định, còn kết quả trống là kết quả hợp lệ chứ không phải lỗi. Khác với Grounding DINO, OMDet-Turbo tách embedding lớp đối tượng khỏi prompt tác vụ ngôn ngữ, vì vậy bước hậu xử lý của `transformers` trả về các nhãn ánh xạ trực tiếp về danh sách lớp đối tượng được truy vấn mà không cần bước phân giải cụm từ.

OMDet-Turbo không có ngưỡng text-token: chỉ `conf` lọc detection, còn truyền `text_threshold` sẽ phát sinh lỗi. Đây là họ duy nhất trong cấp này chạy non-maximum suppression riêng bên trong `post_process_grounded_object_detection`, vì vậy `iou` được tôn trọng ở đây thay vì phát cảnh báo. `imgsz` và `augment=True` bị từ chối hoàn toàn: processor của `transformers` sở hữu bước đổi kích thước, còn tăng cường khi kiểm thử nằm ngoài phạm vi của cấp này. `predict()` trên một ảnh trả về một `Results`, không phải danh sách; truyền thư mục, danh sách ảnh hoặc `stream=True` cho nguồn video để nhận nhiều kết quả. Không có tuyến CLI cho họ mô hình này; `libreyolo predict` chỉ tải checkpoint `.pt` qua `LibreYOLO()`, vì vậy các họ `LibreOpenVocab` chạy từ Python. Xem [dự đoán](/docs/predict) để biết loại nguồn và xử lý luồng.

## Biến thể

Có một checkpoint `t`, kích thước duy nhất của cấp này. Checkpoint mirror `omlab/omdet-turbo-swin-tiny-hf` ở revision upstream cố định qua `OmDetTurboForObjectDetection` của `transformers`; tệp trọng số được mirror giống từng byte với snapshot upstream đó. Chưa có số liệu độ chính xác hoặc độ trễ nào được công bố cho họ mô hình này.

Huấn luyện, xác thực tập dữ liệu (dataset) và xuất đều nằm ngoài phạm vi của cấp này: `train()`, `val()` và `export()` luôn phát sinh `NotImplementedError`. Đây là wrapper chỉ dành cho dự đoán quanh checkpoint đã phát hành.

## Checkpoint

Mọi tệp trọng số đã phát hành cho họ mô hình này.

<checkpoint-table />

## Giấy phép

<provenance-box></provenance-box>

## Trích dẫn

<citation-block />



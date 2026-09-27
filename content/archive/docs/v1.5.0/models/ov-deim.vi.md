---
title: OV-DEIM
families:
  - ov_deim
seo_title: 'OV-DEIM trong LibreYOLO: phát hiện open-vocabulary'
description: >-
  Dùng OV-DEIM trong LibreYOLO để phát hiện open-vocabulary theo kiểu DETR trong
  thời gian thực. Cài extra openvocab và dự đoán với từ vựng văn bản tự do.
lead: >-
  OV-DEIM là detector đối tượng open-vocabulary theo kiểu DETR, ghép các query
  của decoder với embedding văn bản từ text tower MobileCLIP đi kèm. LibreYOLO
  chuyển mô hình sang bản triển khai native dưới dạng họ chỉ dành cho dự đoán
  trong tầng detector open-vocabulary.
keywords:
  - OV-DEIM
  - DEIMv2
  - phát hiện đối tượng open-vocabulary
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

        model = LibreOpenVocab("ov-deim-s")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Thay thế từ vựng
      language: python
      code: >
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE


        model = LibreOpenVocab("ov-deim-l")

        model.set_classes(["traffic light", "bicycle"])

        first = model.predict(SAMPLE_IMAGE, conf=0.3)


        # Lời gọi set_classes() thứ hai thay thế hoàn toàn từ vựng và tạo lại

        # embedding qua text tower; kết quả rỗng là kết quả hợp lệ, không phải
        lỗi.

        model.set_classes(["giraffe"])

        second = model.predict(SAMPLE_IMAGE, conf=0.5)

        print(second.names, len(second))
source_hash: 0c295f555a9eb303
---

## Cài đặt

OV-DEIM được tải qua tầng detector open-vocabulary của LibreYOLO, tầng này cần
extra `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

Không giống phần còn lại của tầng này, OV-DEIM là bản chuyển đổi native của
LibreYOLO chứ không phải wrapper `transformers`; không có lớp mô hình
`transformers` nào cho nó, nhưng cùng extra đó bao gồm các gói `huggingface_hub`,
`safetensors`, `regex` và `ftfy` cần thiết khi dự đoán.

## Dự đoán

OV-DEIM không phải checkpoint mà LibreYOLO tải qua `LibreYOLO()`. Nó được tải
qua factory cùng cấp `LibreOpenVocab`, factory này tải snapshot Hugging Face ở
lần dùng đầu tiên và lưu vào bộ nhớ đệm trong `weights/`.

<code-tabs name="predict" />

`set_classes()` đặt từ vựng văn bản có tính duy trì: gọi lại để thay thế hoàn
toàn danh sách hoặc bỏ qua để giữ các nhãn COCO-80 mặc định; kết quả rỗng là kết
quả hợp lệ, không phải lỗi. Mỗi query của decoder được chấm theo độ tương đồng
cosine với embedding văn bản từ text tower MobileCLIP-B(LT) đi kèm, được tính
trực tuyến cho mọi từ vựng đã đặt và lưu trong bộ nhớ đệm cho đến khi thay đổi.
Vì vậy, prompt tùy ý hoạt động mà không cần tệp embedding tính trước.

OV-DEIM không có ngưỡng token văn bản: chỉ `conf` lọc các phát hiện, còn truyền
`text_threshold` sẽ phát sinh lỗi. Việc ghép là lựa chọn top-K một-một, nên không
có bước non-maximum suppression nào ở đây; `iou` được chấp nhận để tương thích
API nhưng chỉ đưa ra cảnh báo và không làm gì. `imgsz` và `augment=True` bị từ
chối hoàn toàn: mô hình sở hữu đầu vào letterbox cố định và tăng cường dữ liệu
khi kiểm thử nằm ngoài phạm vi tầng này. `predict()` trên một ảnh trả về một
`Results`, không phải danh sách; truyền thư mục, danh sách ảnh hoặc `stream=True`
cho nguồn video để nhận nhiều kết quả. Không có luồng CLI cho họ này,
`libreyolo predict` chỉ tải checkpoint `.pt` qua `LibreYOLO()`, vì vậy các họ
`LibreOpenVocab` chạy từ Python. Xem [dự đoán](/docs/predict) để biết các loại
nguồn và streaming.

Mỗi lời gọi `predict()` cũng chạy text tower MobileCLIP-B(LT) đi kèm để tạo
embedding cho từ vựng hiện tại; xem phần Giấy phép để biết điều này bổ sung gì
vào các điều khoản.

## Các biến thể

Có ba checkpoint là `s`, `m` và `l`. `s` là kích thước mặc định của tầng này khi
không chỉ định. Không giống phần còn lại của tầng, OV-DEIM là bản chuyển đổi
native thay vì wrapper `transformers`: LibreYOLO cung cấp các module detector
theo cùng giấy phép Apache-2.0 như mã thượng nguồn và dùng lại adapter backbone
DINOv3 đã được xây dựng cho họ DEIMv2. Backbone của checkpoint `l` là bản tinh
chỉnh DINOv3-S, được cấp phép riêng theo DINOv3 License của Meta. Chưa có số liệu
độ chính xác hoặc độ trễ nào được công bố cho họ này.

Huấn luyện, đánh giá tập dữ liệu và xuất đều nằm ngoài phạm vi tầng này:
`train()`, `val()` và `export()` luôn phát sinh `NotImplementedError`. Đây là
wrapper chỉ dành cho dự đoán quanh một checkpoint đã công bố.

## Checkpoint

Mọi tệp trọng số đã công bố cho họ này.

<checkpoint-table />

## Giấy phép

<provenance-box>

OV-DEIM áp dụng ba giấy phép thượng nguồn cho mọi lời gọi dự đoán: trọng số
detector theo CC BY-NC 4.0 riêng của OV-DEIM, text tower trực tuyến theo giấy
phép Machine Learning Research Model của Apple (chỉ dùng cho nghiên cứu) và,
đối với checkpoint `l`, bản tinh chỉnh backbone DINOv3-S theo DINOv3 License
của Meta. Văn bản của cả ba giấy phép đều đi kèm trong repo trọng số LibreYOLO.

</provenance-box>

## Trích dẫn

<citation-block />


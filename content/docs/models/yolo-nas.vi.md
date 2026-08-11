---
title: YOLO-NAS
families:
  - yolonas
seo_title: 'YOLO-NAS: dự đoán, huấn luyện và xuất trong LibreYOLO'
description: >-
  Dùng YOLO-NAS trong LibreYOLO để phát hiện và ước lượng tư thế. Trọng số của
  Deci.AI là độc quyền và chỉ dùng phi thương mại; LibreYOLO không công bố trọng
  số nào.
lead: >-
  Một detector tích chập có backbone và neck được tạo ra từ quá trình tìm kiếm
  kiến trúc của Deci.AI, xây dựng bằng các block RepVGG nhận biết lượng tử hóa.
  Trọng số thuộc Deci.AI, chỉ được cấp phép cho mục đích phi thương mại và
  LibreYOLO không công bố trọng số nào.
keywords:
  - YOLO-NAS
  - YOLONAS
  - Deci AI
  - SuperGradients
  - phát hiện đối tượng
  - ước lượng tư thế
  - detector nhận biết lượng tử hóa
  - AutoNAC
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Tên chưa có trên đĩa được lấy từ CDN của Deci. Trước tiên, bản tải

        # in các điều khoản giấy phép của Deci; nhận tệp đồng nghĩa chấp nhận
        chúng.

        model = LibreYOLO("LibreYOLONASs.pt")

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLONASs.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Pose
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Hậu tố -pose chọn head tư thế và tập trọng số riêng.
        model = LibreYOLO("LibreYOLONASs-pose.pt")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLONASs.pt data=my-dataset.yaml \
          epochs=100 imgsz=640 batch=16
    - label: Huấn luyện từ đầu
      language: python
      code: >
        from libreyolo import LibreYOLONAS


        # Không dùng checkpoint Deci nào: mô hình bắt đầu từ trọng số ngẫu
        nhiên,

        # nên kết quả của lượt chạy chỉ bắt nguồn từ dữ liệu của bạn.

        model = LibreYOLONAS(None, size="s")

        model.train(data="my-dataset.yaml", imgsz=640, batch=16)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLONASs.pt data=my-dataset.yaml
    - label: Trên COCO
      language: bash
      code: >
        # YAML COCO đi kèm chứa script tải xuống nhúng sẵn, nên cần quyền rõ
        ràng

        # trừ khi tập dữ liệu đã có cục bộ.

        libreyolo val model=LibreYOLONASl.pt data=coco.yaml imgsz=640 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLONASs.pt format=onnx imgsz=640
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến theo hậu tố tệp, nên artifact đã xuất được tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreYOLONASs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 47c30d6e44024ce7
---

## Cài đặt

YOLO-NAS không cần extra ngoài gói cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Tên checkpoint chưa có trên đĩa được lấy từ CDN công khai của Deci, không phải
từ tổ chức LibreYOLO vì nơi đó không lưu trữ trọng số nào của họ này. Trước khi
truyền tệp, thư viện in các điều khoản giấy phép của Deci một lần cho mỗi tiến
trình; trước khi mở tệp đã tải, SHA-256 của tệp được kiểm tra với giá trị đã
ghim. Những gì các điều khoản cho phép được nêu trong [giấy phép](#licensing).

<code-tabs name="predict" />

Đối tượng `Results` trả về là loại mà mọi họ đều trả về, nên việc đổi sang
detector khác chỉ cần sửa một dòng. `conf` đặt ngưỡng độ tin cậy và `iou` đặt
ngưỡng NMS. Xem [dự đoán](/docs/predict) để biết về nguồn, streaming và xử lý
kết quả.

## Các biến thể

Phát hiện và tư thế dùng cùng kiến trúc với các head khác nhau, đồng thời nhận
cùng các đối số. Các kích thước trong bảng bên dưới dành cho phát hiện; tư thế
được công bố ở các kích thước đó và thêm một kích thước nhỏ hơn. Head tư thế dự
đoán tập keypoint COCO.

<benchmark-table task="detect" />

<va-embed />

## Huấn luyện

<code-tabs name="train" />

Khi bạn bỏ qua `epochs`, `lr0` và `amp`, chúng được phân giải theo từng tác vụ,
vì vậy lượt chạy tư thế bắt đầu với giá trị mặc định khác lượt chạy phát hiện.
Optimizer mặc định là AdamW. Số lớp lấy từ YAML tập dữ liệu và head được xây dựng
lại cho số lớp đó trước epoch đầu tiên; trên head tư thế, số keypoint được xử lý
tương tự, nên checkpoint tư thế COCO có thể tinh chỉnh sang skeleton có kích
thước khác.

Tinh chỉnh bắt đầu từ trọng số của Deci, là đối tượng được giấy phép của Deci
bao quát. Huấn luyện từ mô hình khởi tạo ngẫu nhiên hoàn toàn không liên quan
đến checkpoint Deci, và đó là snippet thứ ba ở trên.

Xem [huấn luyện](/docs/train) để biết về tập dữ liệu, tăng cường dữ liệu, multi-GPU và logger.

## Đánh giá

`val()` trả về từ điển các khóa `metrics/` bao gồm precision, recall, mAP 50 và
mAP 50-95, được đo trên bất kỳ tập dữ liệu nào theo định dạng bạn đã huấn luyện.

<code-tabs name="val" />

## Xuất

<export-matrix />

Artifact đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp
`.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng `Results`. Cũng
hỗ trợ chạy đồ thị trong runtime độc lập không cài LibreYOLO, nhưng khi đó bạn
phải tự viết bước tiền xử lý và hậu xử lý. Mỗi định dạng cài một extra khác nhau
và nhận một số đối số riêng. Cả hai đều được trình bày trên trang của định dạng đó.

Bản xuất là một bản sao khác của cùng trọng số trong container khác. Việc xuất
checkpoint Deci không thay đổi nguồn gốc trọng số hay giấy phép áp dụng cho chúng.

<code-tabs name="export" />

## Checkpoint

Không có tệp nào để liệt kê. Giấy phép của Deci cấm phân phối lại, nên tổ chức
LibreYOLO không công bố trọng số YOLO-NAS và bản tải được phân giải ở nơi khác:
tên có dạng `LibreYOLONAS<size>.pt`, hoặc `LibreYOLONAS<size>-pose.pt` cho tư thế,
ánh xạ đến đối tượng tương ứng trên CDN công khai của Deci.

Chỉ các checkpoint có SHA-256 được thư viện ghim mới có thể lấy theo cách đó.
Mọi tệp khác đều bị từ chối an toàn thay vì mở pickle bên thứ ba chưa xác minh,
và phải được tải thủ công rồi truyền dưới dạng đường dẫn. Tệp đã có trên đĩa
được tải từ đường dẫn, không tải xuống và không qua cổng checksum. Điều này bao
gồm tệp `.pth` của Deci theo tên gốc mà trình tải nhận diện.

## Giấy phép

<provenance-box>

LibreYOLO không lưu trữ hay tạo bản sao các trọng số này: tổ chức Hugging Face
LibreYOLO không có gì cho họ này. Mọi bản tự động tải đều đi đến CDN công khai
của Deci, in các điều khoản của Deci một lần cho mỗi tiến trình trước khi bắt
đầu và được kiểm tra với SHA-256 đã ghim trước khi mở tệp.

Huấn luyện từ mô hình khởi tạo ngẫu nhiên là phương án thay thế. Kiến trúc theo
Apache-2.0 ở thượng nguồn và MIT tại đây, nên mô hình được huấn luyện theo cách
đó trên dữ liệu của bạn không bắt nguồn từ checkpoint Deci nào.

</provenance-box>

## Trích dẫn

YOLO-NAS được phát hành mà không có bài báo. Mục bên dưới là trích dẫn mà tác
giả yêu cầu, bao quát SuperGradients, thư viện phân phối mô hình.

<citation-block />


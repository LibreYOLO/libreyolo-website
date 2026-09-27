---
title: YOLOv9
families:
  - yolo9
seo_title: 'YOLOv9: dự đoán, huấn luyện và xuất theo MIT'
description: >-
  Chạy YOLOv9 trong LibreYOLO, gồm head đầu cuối không NMS và head stride-4 cho
  đối tượng nhỏ. Cài đặt, dự đoán, huấn luyện, đánh giá và xuất.
lead: >-
  Một detector tích chập một giai đoạn: một lượt chấm điểm lưới box dày đặc và
  NMS loại các bản trùng lặp. LibreYOLO cung cấp ba biến thể, trong đó một biến
  thể không có bước NMS.
keywords:
  - YOLOv9
  - YOLO9
  - phát hiện đối tượng
  - phát hiện không NMS
  - phát hiện đầu cuối
  - phát hiện vật thể nhỏ
  - thông tin gradient có thể lập trình
  - GELAN
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Không dùng NMS
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Cùng lời gọi, checkpoint khác. Head đầu cuối tự trả về các dự đoán có
        # điểm cao nhất, nên NMS không chạy và iou bị bỏ qua.
        model = LibreYOLO("LibreYOLO9E2Es.pt")
        result = model(SAMPLE_IMAGE, conf=0.25, max_det=300)

        print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 imgsz=640 batch=16
    - label: Đối tượng nhỏ
      language: python
      code: >
        from libreyolo import LibreYOLO9P2


        # Biến thể stride-4 không có checkpoint COCO riêng, nên hãy chỉ định một

        # checkpoint phát hiện cơ sở: backbone và neck được tải không đổi, còn

        # tower của head stride-4 bắt đầu từ khởi tạo ngẫu nhiên.

        model = LibreYOLO9P2(None, size="s")

        model.train(data="my-dataset.yaml", epochs=100,
        pretrained="LibreYOLO9s.pt")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=my-dataset.yaml
    - label: Trên COCO
      language: bash
      code: >
        # YAML COCO đi kèm chứa script tải xuống nhúng sẵn, nên cần quyền rõ
        ràng

        # trừ khi tập dữ liệu đã có cục bộ.

        libreyolo val model=LibreYOLO9c.pt data=coco.yaml imgsz=640 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640
    - label: Có NMS trong đồ thị
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx nms=True \
          conf=0.25 iou=0.45 max_det=300
    - label: Dùng tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến theo hậu tố tệp, nên artifact đã xuất được tải
        # như mọi checkpoint và trả về cùng đối tượng Results.
        model = LibreYOLO("LibreYOLO9s.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: eaa6023a4a0b9e71
---

## Cài đặt

YOLOv9 không cần extra ngoài gói cơ sở.

```bash
pip install libreyolo
```

## Dự đoán

Trọng số được tải từ Hugging Face ở lần dùng đầu tiên và lưu vào bộ nhớ đệm cục bộ.

<code-tabs name="predict" />

Đối tượng `Results` trả về là loại mà mọi họ đều trả về, nên việc đổi sang
detector khác chỉ cần sửa một dòng. Trên mô hình cơ sở và stride-4, `conf` đặt
ngưỡng độ tin cậy và `iou` đặt ngưỡng NMS. Mô hình đầu cuối không chạy NMS và bỏ
qua `iou`, vì vậy `conf` cùng `max_det` quyết định hình dạng đầu ra. Xem
[dự đoán](/docs/predict) để biết về nguồn, streaming và xử lý kết quả.

## Các biến thể

Ba biến thể dùng chung một backbone. Cả ba chỉ thực hiện phát hiện và nhận cùng các đối số.

Mô hình cơ sở dự đoán trên ba thang feature và loại box trùng lặp bằng NMS.

Mô hình đầu cuối giữ head đó và thêm một nhánh ghép một-một bên cạnh. Suy luận
chỉ đọc nhánh một-một và lấy các dự đoán có điểm cao nhất, nên NMS không chạy.
Chọn mô hình này khi runtime triển khai không có toán tử NMS.

Mô hình stride-4 đưa ra một cấp cao hơn trên backbone, mở rộng neck xuống cấp đó
và dự đoán trên bốn thang thay vì ba. Thang bổ sung dành cho đối tượng chỉ chiếm
ít pixel; checkpoint duy nhất đã công bố cho biến thể này được huấn luyện trên
ảnh chụp từ trên không. Checkpoint phát hiện cơ sở có thể chuyển sang mô hình:
backbone và neck được tải không đổi, ba tower head đã tiền huấn luyện dịch lên
một vị trí, còn tower stride-4 bắt đầu từ khởi tạo ngẫu nhiên.

<benchmark-table task="detect" />

<va-embed />

## Huấn luyện

<code-tabs name="train" />

`pretrained` quyết định điểm bắt đầu của lượt chạy. Truyền `True` để tải
checkpoint đã công bố cho cùng mô hình và kích thước, hoặc truyền tên hay đường
dẫn cho trường hợp khác. Tensor có shape không khớp sẽ bị bỏ qua thay vì bị từ
chối, và lượt chạy ghi lại số lượng đã tải, nên checkpoint được huấn luyện với
số lớp khác vẫn là điểm bắt đầu có thể dùng.

Mô hình stride-4 không có checkpoint COCO riêng đã công bố, nên `True` sẽ phân
giải thành tệp không tồn tại và tải xuống thất bại. Hãy chỉ định checkpoint phát
hiện cơ sở thay thế.

Xem [huấn luyện](/docs/train) để biết về tập dữ liệu, tăng cường dữ liệu, multi-GPU và logger.

## Đánh giá

`val()` trả về từ điển các khóa `metrics/` bao gồm precision, recall, mAP 50 và
mAP 50-95, được đo trên bất kỳ tập dữ liệu nào theo định dạng bạn đã huấn luyện.

<code-tabs name="val" />

## Xuất

<export-matrix />

Dấu kiểm áp dụng cho cả ba biến thể: khi chúng khác nhau, ma trận thể hiện mức hỗ trợ yếu nhất trong ba biến thể.

Artifact đã xuất được tải lại qua `LibreYOLO()` dựa trên hậu tố tệp, vì vậy tệp
`.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng `Results`. Cũng
hỗ trợ chạy đồ thị trong runtime độc lập không cài LibreYOLO, nhưng khi đó bạn
phải tự viết bước tiền xử lý và hậu xử lý.

Với mô hình phát hiện cơ sở, nửa hậu xử lý có thể được đưa vào đồ thị. `nms=True`
trên bản xuất ONNX đặt bước loại bỏ bên trong mô hình, và đầu ra đầu tiên trở thành
tensor cố định `(1, max_det, 6)` có các hàng là `x1, y1, x2, y2, score, class`,
được thêm số 0 sau số lượng phát hiện. Đồ thị đó dùng batch 1 và không có trục
động. Mô hình đầu cuối và stride-4 không chấp nhận cờ này.

Mỗi định dạng cài một extra khác nhau và nhận một số đối số riêng. Cả hai đều được trình bày trên trang của định dạng đó.

<code-tabs name="export" />

## Checkpoint

Mọi tệp trọng số đã công bố cho họ này.

<checkpoint-table />

## Giấy phép

<provenance-box>

Một checkpoint ở đây không theo MIT. Mô hình stride-4 được huấn luyện trên
VisDrone2019-DET kế thừa các điều khoản CC BY-NC-SA 3.0 của tập dữ liệu: chỉ dùng
phi thương mại, mọi sản phẩm phái sinh phải chia sẻ cùng giấy phép, và nằm ngoài
giấy phép dễ dãi áp dụng cho phần còn lại của họ này. Mô hình dự đoán các lớp ảnh
trên không VisDrone thay vì các lớp COCO. Thư viện in toàn bộ thông tin này trước
khi tải tệp.

</provenance-box>

## Trích dẫn

<citation-block />


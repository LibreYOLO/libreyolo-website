---
title: Phát hiện cạnh
seo_title: Phát hiện cạnh trong LibreYOLO
description: >-
  Dự đoán edge-probability map dày đặc từ một ảnh trong LibreYOLO. Chuyển đổi
  checkpoint, đặt ngưỡng cho map, xác thực bằng ODS và OIS, rồi xuất.
lead: >-
  Phát hiện cạnh dự đoán xác suất mỗi pixel nằm trên biên đối tượng. LibreYOLO
  cung cấp dưới dạng tác vụ edge, trả về probability map dày đặc trên canvas ảnh
  gốc thay vì tập đoạn thẳng.
keywords:
  - phát hiện cạnh python
  - boundary detection deep learning
  - edge probability map
  - ODS OIS F-measure
  - dự đoán cạnh dày đặc
last_verified: 1.5.0
snippets:
  predict:
    - label: Dự đoán edge map
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Không có checkpoint edge đi kèm LibreYOLO; hãy chuyển đổi trước (bên
        dưới).

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")

        result = model(SAMPLE_IMAGE, save=True)


        edges = result.edges

        print(edges.array.shape)          # (H, W) float32 trong [0, 1]

        print(edges.binary(0.5).sum())    # số pixel cạnh ở ngưỡng 0.5
    - label: Chọn ngưỡng riêng
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE)

        # Map liên tục được giữ lại để quyết định ngưỡng vẫn thuộc về bạn.
        for t in (0.3, 0.5, 0.7):
            print(t, int(result.edges.binary(t).sum()))
    - label: Lưu ảnh trực quan hóa
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")

        result = model(SAMPLE_IMAGE)


        # plot() render map; phương thức được định nghĩa cho kết quả edge và
        normal.

        result.plot().save("edges.png")
  val:
    - label: Xác thực và đọc các key metric
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=352)

        print(metrics["metrics/ODS"])              # fitness
        print(metrics["metrics/OIS"])
        print(metrics["metrics/best_threshold"])
    - label: Thay đổi lượt quét và dung sai khớp
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(
            data="my-dataset.yaml",
            imgsz=352,
            edge_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5),
            edge_max_dist=0.0075,
        )

        print(metrics["metrics/ODS"], metrics["metrics/best_threshold"])
  export:
    - label: Xuất
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        model.export(format="onnx", imgsz=352)
    - label: Chạy tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến theo hậu tố tệp, vì vậy artifact đã xuất được nạp
        # như mọi checkpoint khác và trả về cùng một đối tượng Results.
        model = LibreYOLO("weights/LibreDexiNedb-edge.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.edges.array.shape)
source_hash: bc286345540ed966
---

## Định nghĩa

Tác vụ `edge` dự đoán một xác suất trên mỗi pixel từ một ảnh RGB: `0` nghĩa là
không phải cạnh và `1` nghĩa là cạnh. Map giữ dạng liên tục, vì vậy lựa chọn
ngưỡng chuyển nó thành ảnh biên nhị phân được để cho bên gọi, còn ngưỡng phù hợp
phụ thuộc vào dataset và mục đích sử dụng tiếp theo.

Dự đoán điền `result.edges`, một payload `EdgeMap` chứa mảng float32 `(H, W)`
trong `[0, 1]` trên canvas ảnh gốc. `.array` trả về map dưới dạng NumPy, còn
`.binary(threshold)` trả về mặt nạ boolean. `result.boxes` luôn rỗng, vì vậy
`conf`, `iou` và `max_det` không có tác dụng. `Results.plot()` hỗ trợ tác vụ này
và render trực tiếp map.

## Mô hình

Ba family phục vụ `edge`.

[DexiNed](/docs/models/dexined), Dense Extreme Inception Network, hợp nhất nhiều
đầu ra bên thành một probability map và chạy ở kích thước gốc 352 px.

[TEED](/docs/models/teed), Tiny and Efficient Edge Detector, là mạng nhỏ ở cùng
kích thước gốc 352 px, với stride downsample 4 so với 16 của DexiNed, vì vậy nó
chấp nhận nhiều giá trị `imgsz` hơn.

[LibreMODUS](/docs/models/libremodus) tạo cạnh kiểu Canny như một target của mô
hình any-to-any. Nó cần thành phần bổ sung `modus` và tài khoản Hugging Face đã
xác thực riêng của bạn, đồng thời không cung cấp `val()` lẫn `export()`, vì vậy
không tham gia các phần xác thực và xuất bên dưới.

## Dự đoán

LibreYOLO không công bố checkpoint edge. Trọng số DexiNed và TEED được phát hành
chính thức được huấn luyện trên BIPED, có điều khoản dataset đã công bố giới hạn
việc sử dụng ở mục đích phi thương mại, vì vậy LibreYOLO không mirror chúng. Hãy
chuyển đổi checkpoint mà bạn được phép sử dụng rồi nạp tệp đã chuyển đổi bằng
đường dẫn:

```bash
python weights/convert_dexined_weights.py upstream.pth weights/LibreDexiNedb-edge.pt --verify
```

<code-tabs name="predict" />

Tên tệp phải có hậu tố tác vụ `-edge` để loader nhận dạng. `imgsz` phải chia hết
cho stride downsample của mạng, và LibreYOLO phát sinh lỗi rõ ràng nêu tên số
chia khi không thỏa mãn. Xem [dự đoán](/docs/predict) để biết về nguồn, stream và
cách xử lý kết quả.

## Định dạng dataset

Xác thực cạnh ghép mỗi ảnh RGB với map một kênh có cùng stem và cùng độ phân
giải, cùng một mặt nạ hợp lệ tùy chọn.

```text
dataset/
  data.yaml
  images/
    val/scene.jpg
  edges/
    val/scene.png
  masks/
    val/scene.png
```

```yaml
path: dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

Target là PNG hoặc TIF một kênh, không phải ảnh trực quan RGB. Map số nguyên
được chia cho giá trị lớn nhất của dtype; map float phải hữu hạn và nằm trong
`[0, 1]`. Pixel mặt nạ được tính là hợp lệ khi khác 0, còn pixel được padding
không bao giờ đóng góp vào metric. `edge_invert: true` hỗ trợ nguồn lưu cạnh đen
trên nền trắng. Xem [định dạng dataset](/docs/reference/dataset-formats) để biết
hợp đồng đầy đủ.

## Huấn luyện

Không edge family nào trong LibreYOLO có implementation huấn luyện: `train()`
phát sinh `NotImplementedError` trên cả ba. Mỗi trang mô hình nêu tên script
chuyển đổi checkpoint được huấn luyện ở nơi khác thành dạng LibreYOLO có thể nạp.

## Xác thực

`val()` báo cáo F-measure kiểu BSDS. Dự đoán liên tục trước hết được làm mảnh
bằng non-maximum suppression gradient bốn hướng, sau đó pixel cạnh dự đoán và
ground truth được khớp một-một trong dung sai khoảng cách.

<code-tabs name="val" />

`metrics/ODS` là F-measure optimal-dataset-scale: số lượng khớp được gộp trên
dataset tại từng ngưỡng và F-measure tốt nhất trong các giá trị gộp đó được báo
cáo. Nó cũng là `fitness`, con số cơ chế chọn checkpoint tốt nhất đọc.
`metrics/OIS` là F-measure optimal-image-scale, trung bình trên các ảnh của
F-measure tốt nhất riêng cho từng ảnh, vì vậy cho phép mỗi ảnh chọn ngưỡng riêng.
`metrics/best_threshold` là ngưỡng duy nhất tạo ra ODS, cũng là ngưỡng nên dùng
lại trong `edges.binary()` khi inference.

Hai đối số định hình lượt quét. `edge_thresholds` là tập ngưỡng được thử, mặc
định từ 0.01 đến 0.99 theo bước một phần trăm. `edge_max_dist` là dung sai khớp
dưới dạng tỷ lệ đường chéo ảnh, mặc định `0.0075`; cặp xa hơn mức đó không khớp.

## Xuất

Mô hình cạnh đã xuất được nạp lại qua `LibreYOLO()` theo hậu tố tệp, vì vậy tệp
`.onnx` hoạt động như checkpoint và trả về cùng `Results`.

<code-tabs name="export" />

Việc xuất cạnh dùng hợp đồng runtime batch 1, độ phân giải cố định: `dynamic` và
`batch` khác 1 bị từ chối, còn graph đã xuất phát ra một probability map hợp
nhất. Phạm vi theo định dạng nằm trên các trang
[DexiNed](/docs/models/dexined), [TEED](/docs/models/teed) và trong [ma trận xuất
đầy đủ](/docs/reference/export-matrix). Phần [Xuất](/docs/export) liệt kê các đối
số mà mọi định dạng chấp nhận.

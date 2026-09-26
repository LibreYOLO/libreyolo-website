---
title: Ước lượng độ sâu
seo_title: Ước lượng độ sâu monocular trong LibreYOLO
description: >-
  Dự đoán depth map tương đối dày đặc từ một ảnh trong LibreYOLO. So sánh các
  depth family, đọc metric độ sâu và xuất mô hình độ sâu.
lead: >-
  Ước lượng độ sâu dự đoán khoảng cách từ mỗi pixel đến camera chỉ bằng một ảnh.
  LibreYOLO cung cấp dưới dạng tác vụ depth, trả về inverse-depth map tương đối
  dày đặc trên canvas ảnh gốc.
keywords:
  - ước lượng độ sâu monocular python
  - tạo depth map từ một ảnh
  - mô hình độ sâu tương đối
  - depth anything libreyolo
  - dự đoán độ sâu dày đặc
last_verified: 1.5.0
snippets:
  predict:
    - label: Dự đoán depth map
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.data.shape)              # (H, W) trên canvas gốc
        print(depth.min, depth.max, depth.mean)
    - label: Làm việc với các giá trị
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")

        result = model(SAMPLE_IMAGE)


        depth = result.depth_map

        raw = depth.data          # cao hơn là gần hơn; không đơn vị thực, không
        tỷ lệ

        gray = depth.normalized() # đổi tỷ lệ về [0, 1] để trực quan hóa

        print(raw.shape, float(gray.max()))
    - label: Lựa chọn nhỏ gọn
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Cùng hợp đồng tác vụ, mạng nhỏ hơn nhiều dành cho runtime biên.
        model = LibreYOLO("LibreZipDepthb-depth.pt")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
  val:
    - label: Xác thực và đọc các key metric
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])   # fitness
        print(metrics["metrics/delta2"], metrics["metrics/delta3"])
  export:
    - label: Xuất
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        model.export(format="onnx")
    - label: Chạy tệp đã xuất
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory định tuyến theo hậu tố tệp, vì vậy artifact đã xuất được nạp
        # như mọi checkpoint khác và trả về cùng một đối tượng Results.
        model = LibreYOLO("LibreDepthAnythingV2s-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: e0612c59f9c999b4
---

## Định nghĩa

Tác vụ `depth` dự đoán một giá trị trên mỗi pixel từ một ảnh RGB. LibreYOLO định
nghĩa giá trị đó là inverse depth tương đối: giá trị cao hơn nghĩa là gần camera
hơn, còn các số không mang đơn vị thực và không có tỷ lệ giữ nguyên giữa hai
ảnh. So sánh độ sâu giữa hai pixel trong cùng dự đoán có ý nghĩa; so sánh một
giá trị với giá trị từ ảnh khác thì không.

Dự đoán điền `result.depth_map`, một payload `DepthMap` chứa mảng `(H, W)` trên
canvas ảnh gốc. `.min`, `.max` và `.mean` đọc các giá trị hữu hạn, còn
`.normalized()` đổi tỷ lệ map về `[0, 1]` để hiển thị. `result.boxes` luôn rỗng,
vì vậy `conf`, `iou` và `max_det` không có tác dụng, còn `save=True` ghi ảnh đã
áp colormap của map thay vì ảnh có chú thích.

## Mô hình

Sáu family phục vụ `depth`.

[Depth Anything V2](/docs/models/depth-anything-v2) ghép encoder DINOv2 với
decoder DPT và là lựa chọn mặc định đa dụng tại đây. Giấy phép ảnh hưởng tới
việc chọn kích thước nhiều như độ chính xác: checkpoint Small dùng Apache-2.0,
còn Base và Large chỉ cho mục đích phi thương mại, vì vậy hãy kiểm tra bảng
checkpoint trên trang mô hình trước khi chọn.

[Depth Anything 3](/docs/models/depth-anything-3) port checkpoint DA3MONO-LARGE,
một transformer thuần không có chuyên biệt kiến trúc cho độ sâu.

[ZipDepth](/docs/models/zipdepth) là tầng nhỏ gọn: CNN có thể tái tham số hóa,
được chưng cất từ Depth Anything V2 Large, với checkpoint thứ hai có decoder
tránh phép toán gather và unfold cho NPU compiler không hỗ trợ chúng.

[MiDaS](/docs/models/midas) là dòng công trình đã thiết lập giao thức độ sâu
tương đối zero-shot dùng để đo các family khác. Đây là depth family duy nhất mà
LibreYOLO không công bố lại: yêu cầu checkpoint sẽ tải asset chính thức từ bản
phát hành GitHub của tác giả và kiểm tra SHA-256 cố định.

[LibreMODUS](/docs/models/libremodus) thực hiện độ sâu như một target của mô hình
any-to-any thay vì head chuyên dụng. Nó cần thành phần bổ sung `modus` và tài
khoản Hugging Face đã xác thực riêng của bạn, đồng thời không cung cấp `val()`
lẫn `export()`.

[SenseNova-Vision](/docs/models/sensenova-vision) tạo depth map như một ảnh qua
diffusion decode, từ cùng checkpoint 7B phục vụ sáu tác vụ khác. Nó cần thành
phần bổ sung `sensenova`, còn trọng số bị giới hạn cho mục đích phi thương mại;
giấy phép nằm trên trang của mô hình.

## Dự đoán

Trọng số được tải từ Hugging Face trong lần sử dụng đầu tiên và lưu vào cache
cục bộ, trừ hai family đã nêu ở trên.

<code-tabs name="predict" />

Độ phân giải đầu vào bị ràng buộc theo từng family. Depth Anything V2 và Depth
Anything 3 dựa trên patch grid DINOv2, vì vậy `imgsz` phải chia hết cho 14, điều
LibreYOLO kiểm tra trước khi chạy. `Results.plot()` không hỗ trợ tác vụ này;
phương thức chỉ được định nghĩa cho pháp tuyến bề mặt và cạnh. Xem [dự
đoán](/docs/predict) để biết về nguồn, stream và cách xử lý kết quả.

## Định dạng dataset

Xác thực độ sâu ghép mỗi ảnh với depth map một kênh dày đặc có cùng độ phân giải,
được tìm bằng cách thay thư mục độ sâu vào đường dẫn ảnh.

```text
dataset/
  data.yaml
  images/
    val/room.jpg
  depths/
    val/room.png
```

```yaml
path: dataset
val: images/val
depths_dir: depths
nc: 1
names: {0: depth}
```

Map là PNG hoặc TIF một kênh, hoặc `.npy`. Giá trị là độ sâu thuần theo đơn vị
mà dataset giữ nhất quán, còn pixel `0`, âm, NaN và vô cực đánh dấu sample không
hợp lệ bị loại khỏi metric. Map số nguyên được chia cho `depth_scale`, mặc định
là `256.0`, quy ước PNG 16-bit; map float `.npy` được dùng nguyên trạng.
`depth_stem_suffix` và `depth_mask_suffix` hỗ trợ dataset đặt tên tệp độ sâu hoặc
mặt nạ hợp lệ theo cách khác. Xem [định dạng
dataset](/docs/reference/dataset-formats) để biết hợp đồng đầy đủ.

## Huấn luyện

Không depth family nào trong LibreYOLO có implementation huấn luyện: `train()`
phát sinh `NotImplementedError` trên cả sáu. Mỗi trang mô hình nêu tên script
chuyển đổi checkpoint được huấn luyện ở upstream thành dạng LibreYOLO có thể nạp.

## Xác thực

`val()` chạy validator độ sâu dùng chung. Độ sâu tương đối không có tỷ lệ tuyệt
đối, vì vậy mỗi dự đoán trước hết được fit vào nghịch đảo ground truth bằng tỷ
lệ và độ dịch chuyển least-squares theo từng ảnh, rồi được đảo lại thành độ sâu.
Mọi metric bên dưới được tính theo từng ảnh trên map đã căn chỉnh đó và lấy
trung bình trên dataset, chỉ tính các pixel được dataset đánh dấu hợp lệ.

<code-tabs name="val" />

`metrics/abs_rel` là sai số tương đối tuyệt đối trung bình, residual chia cho độ
sâu ground truth, giá trị thấp hơn là tốt hơn. `metrics/rmse` là căn bậc hai sai
số bình phương trung bình theo đơn vị độ sâu riêng của dataset, cũng thấp hơn là
tốt hơn. `metrics/delta1`, `metrics/delta2` và `metrics/delta3` là độ chính xác
theo ngưỡng: tỷ lệ pixel hợp lệ có tỷ số với ground truth, lấy chiều nào lớn hơn,
nằm dưới 1.25, bình phương 1.25 và lập phương 1.25, vì vậy cao hơn là tốt hơn.
`metrics/delta1` cũng là `fitness`, con số mà cơ chế chọn checkpoint tốt nhất đọc.

## Xuất

Mô hình độ sâu đã xuất được nạp lại qua `LibreYOLO()` theo hậu tố tệp, vì vậy
tệp `.onnx` hoặc `.engine` hoạt động như checkpoint và trả về cùng `Results`,
với `depth_map` thay cho hộp.

<code-tabs name="export" />

Phạm vi hỗ trợ khác nhau theo family, còn Depth Anything 3 từ chối mọi định dạng
ngoài tập đã xác thực thay vì thử chuyển đổi chưa được xác thực. Kiểm tra trang
mô hình và [ma trận xuất đầy đủ](/docs/reference/export-matrix) trước khi cam
kết target. LibreMODUS và SenseNova-Vision hoàn toàn không xuất được. Phần
[Xuất](/docs/export) liệt kê các đối số mà mọi định dạng chấp nhận.

---
title: Các kiểu Results
seo_title: Tham chiếu object Results của LibreYOLO
description: >-
  Mọi payload mà object Results LibreYOLO có thể chứa, một slot cho mỗi dạng tác
  vụ: box, mask, keypoint, probs, obb, depth, ocr, embedding và mười loại khác.
lead: >-
  Results là kiểu trả về theo ảnh duy nhất của mọi mô hình LibreYOLO. Nó có mười
  tám slot payload tùy chọn, mỗi slot ứng với một dạng tác vụ và chỉ điền những
  slot mô hình tạo ra.
keywords:
  - object results libreyolo
  - Results.boxes
  - Results.masks
  - Results.probs
  - Results.depth_map
  - Results.summary
  - libreyolo results to_json
last_verified: 1.5.0
verification: >-
  Tên slot, shape, thuộc tính và giá trị mặc định được đọc từ
  libreyolo/utils/results.py ở v1.5.0. Ngữ nghĩa được lấy từ docstring của các
  lớp payload.
snippets:
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE)

        print(result.orig_shape, result.path)
        print(result.boxes.xyxy)
        print(result.boxes.conf)
        print(result.names[int(result.boxes.cls[0])])
  convert:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE)

        # Mọi payload được di chuyển cùng nhau.
        result = result.cpu().numpy()

        # Các hàng dưới dạng dict thuần, sau đó là JSON.
        print(result.summary()[:1])
        print(result.to_json())
source_hash: 16f654364ae6448a
---

## Object Results

Một `Results` mô tả một ảnh. Source là một ảnh trả về một object, source dạng
danh sách hoặc thư mục trả về danh sách, còn `stream=True` trả về generator tạo
lần lượt từng object.

| Thuộc tính | Kiểu | Ý nghĩa |
|---|---|---|
| `orig_shape` | `(int, int)` | Chiều cao và chiều rộng ảnh gốc |
| `path` | `str` | Đường dẫn nguồn khi đầu vào đến từ ổ đĩa |
| `names` | `dict[int, str]` | Ánh xạ chỉ mục lớp sang tên lớp |
| `speed` | `dict[str, float]` | Số mili giây theo từng giai đoạn |
| `track_id` | tensor | Track ID khi kết quả đến từ `track()` |
| `frame_idx` | `int` | Chỉ mục frame cho source video và stream |
| `restore_scale` | `int` | Hệ số upscale đầu ra so với đầu vào của kết quả restore; `1` ở mọi nơi khác |

<code-tabs name="usage" />

## Slot payload

Mỗi slot là `None` trừ khi mô hình tạo ra nó. Tác vụ của họ quyết định slot được
điền.

| Slot | Lớp | Tác vụ |
|---|---|---|
| `boxes` | `Boxes` | detect |
| `masks` | `Masks` | segment |
| `keypoints` | `Keypoints` | pose |
| `probs` | `Probs` | classify |
| `obb` | `OBB` | obb |
| `gaze` | `Gaze` | gaze |
| `points` | `Points` | point |
| `semantic_mask` | `SemanticMask` | semantic |
| `panoptic` | `PanopticSegmentation` | panoptic |
| `depth_map` | `DepthMap` | depth |
| `normal_map` | `NormalMap` | normal |
| `edges` | `EdgeMap` | edge |
| `restored` | `RestoredImage` | restore |
| `matte` | `Matte` | matte |
| `ocr` | `OCRRegions` | ocr |
| `embeddings` | `Embeddings` | embed |
| `identities` | `Identities` | embed, kèm gallery |
| `meshes` | `Meshes` | mesh |

`result.normals` là alias đọc ghi của `result.normal_map`.

Có thể đặt nhiều slot cùng lúc. Mô hình segmentation điền cả `boxes` và
`masks`; mô hình gaze điền `boxes` bằng box khuôn mặt và `gaze` bằng các góc;
mô hình mesh điền `boxes` bằng box người và `meshes` căn theo hàng với chúng.

## Boxes

Box detection cho một ảnh.

| Thành viên | Trả về |
|---|---|
| `xyxy` | Tọa độ góc theo pixel ảnh gốc |
| `xywh` | Tâm và kích thước theo pixel |
| `xyxyn` | Góc chuẩn hóa vào `[0, 1]` |
| `xywhn` | Tâm và kích thước chuẩn hóa vào `[0, 1]` |
| `conf` | Độ tin cậy theo box |
| `cls` | Chỉ mục lớp theo box |
| `id` | Track ID theo box hoặc `None` |
| `is_track` | `True` khi có track ID |
| `data` | Tensor đã đóng gói |

`with_id(id)` và `with_orig_shape(orig_shape)` trả về `Boxes` mới có trường đó
được thay thế.

## Masks

Mask instance cho một ảnh. `data` là tensor mask; `xy` trả về contour theo
instance bằng pixel, còn `xyn` trả về dạng chuẩn hóa.

## Keypoints

Keypoint pose được căn theo hàng với `boxes`. `xy` là cặp tọa độ theo keypoint,
`xyn` là cặp đã chuẩn hóa. `conf` là kênh thứ ba khi dữ liệu có kênh này, nếu
không là `None`. `has_visible` là mảng boolean, true tại nơi `conf > 0`, và
toàn true khi không có kênh độ tin cậy.

## Points

Định vị điểm cho một ảnh. `data` có shape `(N, 4)` với hàng `x, y, class,
confidence`. Tọa độ là pixel tuyệt đối; `xy`, `cls` và `conf` tách các cột, còn
`xyn` chuẩn hóa tọa độ.

## Probs

Score classification. `top1` là chỉ mục thắng, `top5` là năm chỉ mục tốt nhất,
còn `top1conf` và `top5conf` là các score tương ứng.

## OBB

Box có hướng. `data` chứa 7 hoặc 8 giá trị mỗi hàng: `xywhr`, track ID tùy chọn,
sau đó là độ tin cậy và lớp.

| Thành viên | Trả về |
|---|---|
| `xywhr` | Tâm, kích thước và góc xoay theo radian |
| `xyxyxyxy` | Bốn góc theo pixel |
| `xyxyxyxyn` | Bốn góc đã chuẩn hóa |
| `xyxy` | Bao thẳng trục theo pixel |
| `conf`, `cls`, `id`, `is_track` | Như trên `Boxes` |

## Gaze

Góc nhìn theo khuôn mặt bằng radian, shape `(N, 2)`, căn theo hàng với box khuôn
mặt trong `boxes`. Cột 0 là pitch và cột 1 là yaw theo quy ước L2CS: yaw dương
xoay hướng nhìn về bên trái của đối tượng, còn pitch dương xoay xuống dưới.
`pitch_deg` và `yaw_deg` chuyển sang độ, còn `direction_3d` trả về vector hướng
đơn vị.

## SemanticMask

Bản đồ semantic dày đặc, shape `(H, W)` gồm ID lớp số nguyên trên canvas ảnh
gốc. `255` là giá trị bỏ qua và không bao giờ tính là lớp
(`SemanticMask.IGNORE_INDEX`). `classes` liệt kê ID lớp hiện diện, còn
`class_mask(class_id)` trả về mask boolean cho một lớp.

## PanopticSegmentation

Mỗi pixel nhận đúng một segment không chồng lấp, hợp nhất vùng stuff và instance
thing. `data` là bản đồ ID segment số nguyên `(H, W)`; ID segment `0` là chưa
gán nhãn (`PanopticSegmentation.IGNORE_INDEX`). `segments_info` là danh sách
dict, mỗi segment một mục, có ít nhất `{"id": int, "category_id": int}`, trong
đó `id` khớp một giá trị trong bản đồ và `category_id` lập chỉ mục `names`.
`segment_ids` liệt kê ID hiện diện, còn `segment_mask(segment_id)` trả về mask
boolean của một segment.

Phân biệt thing với stuff là thuộc tính của category chứ không phải segment.
Payload có thể phi chuẩn hóa thuộc tính này lên từng segment dưới dạng
`"isthing": bool`, và khi có, giá trị phải khớp map cấp category.

## DepthMap

Bản đồ độ sâu nghịch đảo tương đối dày đặc, shape `(H, W)` gồm số float trên
canvas ảnh gốc. Giá trị cao hơn nghĩa là gần camera hơn. Giá trị mang tính tương
đối, không phải mét theo hệ metric. `min`, `max` và `mean` được tính trên các
giá trị hữu hạn, còn `normalized()` rescale bản đồ vào `[0, 1]`.

## NormalMap

Trường pháp tuyến bề mặt dày đặc, float32 `(H, W, 3)` trên canvas ảnh gốc, theo
hệ camera OpenCV: `+x` sang phải, `+y` xuống dưới, `+z` đi vào cảnh. Pháp tuyến
hướng về camera nên bề mặt song song mặt phẳng ảnh là `(0, 0, -1)`. Mỗi pixel
là vector đơn vị. `assert_normalized(atol=1e-4)` kiểm tra bất biến này.

## EdgeMap

Bản đồ xác suất cạnh dày đặc, float32 `(H, W)` trên canvas ảnh gốc, trong đó `0`
là không phải cạnh và `1` là cạnh. Bản đồ liên tục được giữ để người gọi tự chọn
ngưỡng: `binary(threshold=0.5)` áp dụng ngưỡng, còn `array` trả về view numpy.

## RestoredImage

Ảnh RGB đã khôi phục, uint8 `(H, W, 3)`. Với super-resolution, canvas gấp
`Results.restore_scale` lần đầu vào. `array` trả về view numpy và `save(path)`
ghi ảnh.

## Matte

Matte độ mờ mềm, float32 `(H, W)` trong `[0, 1]` trên canvas ảnh gốc. `1` là
hoàn toàn tiền cảnh và `0` là hoàn toàn nền. Matte mềm bao hàm mask xóa nền
cứng được lấy ngưỡng ở 0.5, đồng thời giữ cạnh anti-aliased mà mask nhị phân bỏ
mất. `array` trả về view numpy.

Trên kết quả matte, `Results.cutout(image=None)` trả về mảng RGBA uint8 `(H, W,
4)` có kênh thứ tư là matte, còn `Results.save(path, image=None)` ghi cutout đó
thành PNG nền trong suốt. Cả hai lấy RGB từ `image` khi được cung cấp, nếu không
sẽ nạp lại từ `Results.path`.

## OCRRegions

Văn bản đã định vị cùng nội dung nhận dạng. `data` là polygon float `(N, 4, 2)`
theo pixel ảnh gốc, thứ tự trên-trái, trên-phải, dưới-phải, dưới-trái; các vùng
theo thứ tự đọc từ trên xuống dưới rồi từ trái sang phải. `texts` là danh sách
N nội dung. `conf` là score nhận dạng theo vùng và `det_conf` là score detection,
cả hai có shape `(N,)`.

Tứ giác detection là polygon thực nên không điền `Results.boxes`. `xyxy` cung
cấp bao thẳng trục.

## Embeddings

Vector chuẩn hóa L2 từ tác vụ `embed`, luôn có shape `(N, D)`. Kết quả toàn ảnh
có một hàng và không có box; embedding theo vùng căn theo hàng với `boxes`. Vì
mỗi hàng được chuẩn hóa, cosine similarity là tích vô hướng.

| Thành viên | Trả về |
|---|---|
| `dim` | `D` |
| `normalized` | Các hàng được chuẩn hóa lại |
| `similarity(other)` | Cosine similarity từng cặp với `Embeddings` hoặc tensor khác |
| `verify(i, j, threshold=0.4)` | `True` khi hàng `i` và `j` khớp |

## Identities

Kết quả khớp gallery có tên, căn theo hàng với `embeddings`. Được tạo khi truyền
`Gallery` cho dự đoán `embed`. `name` là danh sách có mục bằng `None` khi dưới
ngưỡng khớp, và tên gần nhất dưới ngưỡng không bao giờ được đoán. `score` là
mảng score khớp, còn `data` ghép chúng thành cặp.

## Meshes

Mesh cơ thể người tham số hóa, căn theo hàng với box người trong `boxes`. Mọi
thứ nằm trong hệ camera của ảnh gốc. `transl` theo mét với `+z` hướng ra xa
camera; `vertices` và `joints3d` theo metric và đã bao gồm `transl`; `joints2d`
theo pixel trên canvas ảnh gốc chứ không phải crop mạng đã thấy. Không trường
nào mang hệ tọa độ thế giới hoặc trọng lực.

Layout tham số khác nhau giữa các mô hình cơ thể nên không shape nào được
hardcode. `body_model` nêu tên phép tham số hóa, còn số lượng được đọc lại từ
tensor: `num_vertices`, `num_joints`, `num_betas` và `has_vertices`. `params`
trả về dictionary tham số, còn `save_obj(path, index=0)` ghi một mesh. Các
trường gồm `global_orient`, `body_pose`, `betas`, `transl`, `vertices`, `faces`,
`joints3d`, `joints2d`, `conf`, `focal_length` và `extras`.

Với `body_model="mhr"`, rotation là góc Euler theo radian thay vì axis-angle,
`body_pose` là vector tham số phẳng theo khớp thay vì một bộ ba cho mỗi khớp,
còn `betas` là hệ số identity blendshape. Scale skeleton, pose bàn tay và biểu
cảm khuôn mặt nằm trong `extras`.

## Chuyển đổi và lựa chọn

Mọi payload có `to(*args, **kwargs)`, `cpu()`, `cuda()` và `numpy()`, và gọi một
trong số đó trên `Results` sẽ áp dụng cho mọi slot được điền cùng lúc.

<code-tabs name="convert" />

`result[idx]` chọn hàng trên các payload căn theo hàng. `len(result)` là số
detection hoặc số điểm khi không có box. `result.update(...)` trả về bản sao có
các slot được nêu thay thế; nó nhận mọi slot cùng `track_id` và `restore_scale`.

## summary và to_json

`summary(normalize=False, decimals=5, embeddings=False)` trả về danh sách dict
thuần, mỗi detection, segment, điểm hoặc vùng một hàng tùy các slot đã đặt.
`to_json(**kwargs)` chuyển đối số đến `summary` và trả về chuỗi JSON.

`plot()` render kết quả pháp tuyến dense hoặc cạnh trong biểu diễn chuẩn; nó báo
lỗi với loại kết quả khác. Ảnh chú thích cho các tác vụ còn lại đến từ
`predict(save=True)`.

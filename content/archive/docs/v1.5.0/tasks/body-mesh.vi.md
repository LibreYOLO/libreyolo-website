---
title: Lưới cơ thể
seo_title: Khôi phục lưới cơ thể trong LibreYOLO
description: >-
  Khôi phục lưới cơ thể 3D tham số cho mỗi người trong LibreYOLO. Dự đoán từ hộp
  người hoặc detector, rồi đọc vertex, khớp và độ dịch chuyển camera.
lead: >-
  Khôi phục lưới cơ thể biến một ảnh và tập hộp người thành cơ thể 3D tham số
  cho từng người: tham số hình dạng và tư thế, vertex đã tạo tư thế, khớp 3D và
  độ dịch chuyển camera đặt chúng phía trước ống kính.
keywords:
  - khôi phục human mesh python
  - lưới cơ thể
  - tư thế cơ thể 3d
  - SAM 3D Body
  - MHR
  - mô hình cơ thể tham số
  - tác vụ mesh libreyolo
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import SAMPLE_IMAGE

        from libreyolo.models.sam3dbody import LibreSAM3DBody


        # Family này không được đăng ký với factory LibreYOLO(), vì vậy phải

        # dựng trực tiếp. model_path=None kích hoạt lượt tải Hugging Face được

        # kiểm soát truy cập; chuỗi được xử lý như checkpoint cục bộ hiện có

        # và không bao giờ được tải. Inference cần CUDA.

        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])


        meshes = result.meshes

        print(meshes.body_model)      # cách tham số hóa mà các tensor này sử
        dụng

        print(meshes.vertices.shape)  # (N, V, 3), hệ tọa độ camera, mét

        print(meshes.joints3d.shape)  # (N, J, 3)

        print(meshes.joints2d.shape)  # (N, J, 2), pixel trên ảnh nguồn
    - label: Với detector người
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # person_detector nhận detector LibreYOLO đã dựng, callable thuần hoặc
        # instance PersonDetector. Không có tên rút gọn.
        detector = LibreYOLO("LibreYOLO9s.pt")
        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_detector=detector)
source_hash: 31c5b44171cbcd0e
---

## Định nghĩa

Khôi phục lưới cơ thể trả về một payload `Meshes` cho mỗi ảnh, với các dòng được
căn theo `result.boxes`: dòng `i` mô tả người trong hộp `i`, cùng hợp đồng mà
tác vụ tư thế dùng cho keypoint.

Mọi thứ được biểu diễn trong hệ tọa độ camera của ảnh gốc. `transl` dùng đơn vị
thực là mét, với +z hướng ra xa camera. `vertices` và `joints3d` dùng đơn vị thực
và đã bao gồm `transl`, vì vậy không cần kết hợp thêm. `joints2d` có đơn vị pixel
trên canvas ảnh gốc, không phải trên vùng crop mà mạng đã thấy. `faces` chứa
topology của lưới một lần cho toàn ảnh thay vì theo từng dòng vì mọi người dùng
chung topology. Phiên bản này không có hệ tọa độ thế giới hoặc trọng lực, và
không trường nào âm thầm thay thế cho chúng.

Bố cục tham số khác nhau giữa các mô hình cơ thể, vì vậy không shape nào cố
định: `body_model` đặt tên cách tham số hóa và số lượng được đọc lại từ tensor.
Với `"mhr"`, Momentum Human Rig, phép xoay là góc Euler theo radian thay vì
axis-angle, `body_pose` là vector tham số phẳng theo từng khớp thay vì một bộ ba
cho mỗi khớp, còn `betas` là hệ số identity blendshape. Tỷ lệ skeleton, tư thế
bàn tay và biểu cảm khuôn mặt nằm trong `extras`.

Key tác vụ chuẩn là `mesh`. `body-mesh`, `hmr` và `human-mesh-recovery` được
chuẩn hóa về key đó.

## Mô hình

[SAM 3D Body](/docs/models/sam-3d-body) là family duy nhất phục vụ tác vụ này,
và đây là wrapper thay vì bản port: package `sam-3d-body` của Meta được công bố
theo SAM License mà mã nguồn riêng của LibreYOLO không được phép phái sinh, vì
vậy không phần nào được đưa trực tiếp vào. Hai backbone dùng chung mô hình cơ
thể MHR, `d3` trên encoder DINOv3 ViT-H/16+ và `h` trên ViT-H gốc.

Có ba yêu cầu trước dự đoán đầu tiên, và không yêu cầu nào là tùy chọn.

Bạn tự cài package upstream, không phải LibreYOLO:

```bash
git clone https://github.com/facebookresearch/sam-3d-body
pip install roma einops yacs omegaconf braceexpand pytorch-lightning timm
```

Trỏ thư viện tới bản clone bằng `sam_3d_body_path=` hoặc biến môi trường
`SAM_3D_BODY_PATH`. Người dùng không bao giờ dựng family này sẽ không bao giờ
kích hoạt import.

Kho mirror checkpoint được kiểm soát truy cập. Hãy chấp nhận giấy phép trên
trang mô hình Hugging Face và xác thực bằng `hf auth login`, nếu không lượt tải
đầu tiên sẽ thất bại. Bản thân mô hình cơ thể MHR là một bản phát hành
Apache-2.0 riêng, được tải từ vị trí công khai riêng và lưu vào cache cục bộ.

Inference cần thiết bị CUDA. Estimator upstream di chuyển batch lên GPU mà
không kiểm tra, vì vậy không có đường dẫn CPU để quay về và `device="cpu"` sẽ
phát sinh lỗi.

## Dự đoán

<code-tabs name="predict" />

Người được đưa tới mô hình theo một trong hai cách. `person_boxes` truyền các
hộp bạn đã có, chỉ cho một ảnh: tập hộp cố định không thể đi theo người giữa các
frame video, nên dùng với nguồn video sẽ phát sinh lỗi thay vì âm thầm sử dụng
lại hộp của frame đầu tiên. `person_detector` nhận detector LibreYOLO đã dựng,
callable hoặc `PersonDetector`, và là đường dẫn cho video. `focal_length` cung
cấp thông số nội tại camera đã biết; nếu không đặt, mô hình dùng giá trị ước
lượng riêng, cũng là giá trị `meshes.focal_length` báo cáo.

Family này không được nối vào factory `LibreYOLO()` hoặc lệnh CLI
`libreyolo predict`. `LibreSAM3DBody` là điểm vào duy nhất. Xem [dự
đoán](/docs/predict) để biết về nguồn, stream và cách xử lý kết quả.

## Huấn luyện

Không family nào trong tác vụ này huấn luyện bên trong LibreYOLO.
`LibreSAM3DBody.train()` phát sinh lỗi: hãy huấn luyện tại dự án upstream và nạp
checkpoint kết quả tại đây.

## Xác thực

Không có validator lưới và `val()` phát sinh lỗi. Các benchmark thông thường
chỉ dành cho giấy phép nghiên cứu, vì vậy không benchmark nào đi kèm và không
thể tự động tải cho bạn.

Bản thân metric có sẵn dưới dạng `libreyolo.validation.mesh_metrics` để đánh
giá trên dataset bạn đã có. Hàm nhận khớp dự đoán và target, tùy chọn nhận vertex
dự đoán và target, rồi trả về dictionary có key chính xác như validator:

`metrics/mpjpe` là sai số vị trí trung bình trên mỗi khớp sau khi căn chỉnh khớp
gốc, vì vậy nó tính điểm tư thế mà bỏ qua vị trí đứng của người trong cảnh.
`metrics/pa_mpjpe` là cùng đại lượng sau khi căn chỉnh Procrustes đầy đủ gồm xoay,
tỷ lệ đồng nhất và dịch chuyển, loại bỏ lỗi hướng toàn cục cùng kích thước cơ thể
và chỉ để lại tư thế khớp. `metrics/pve` là sai số trung bình trên mỗi vertex
trên bề mặt lưới sau khi căn theo tâm vertex; khác với metric khớp, nó nhạy với
hình dạng cơ thể và chỉ xuất hiện khi cung cấp cả hai mảng vertex. Giá trị thấp
hơn là tốt hơn cho cả ba. Đầu vào được giả định dùng đơn vị thực là mét, còn
`scale_to_mm` chuyển kết quả sang mili mét thường được báo cáo trong tài liệu.

## Xuất

Chưa triển khai xuất lưới. LibreYOLO chưa định nghĩa hợp đồng metadata graph đã
xuất cho tác vụ này, gồm cách mang bố cục tham số MHR ra ngoài PyTorch, vì vậy
`export()` phát sinh lỗi thay vì phát ra graph có đầu ra không thể diễn giải.

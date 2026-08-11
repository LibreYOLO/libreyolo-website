---
title: SAM 3D Body
families:
  - sam3dbody
seo_title: 'SAM 3D Body: khôi phục mesh toàn thân trong LibreYOLO'
description: >-
  Dùng SAM 3D Body trong LibreYOLO để khôi phục mesh toàn thân người. Cài đặt và
  dự đoán; các checkpoint bị giới hạn truy cập theo SAM License của Meta và yêu
  cầu CUDA.
lead: >-
  SAM 3D Body là mô hình dùng prompt của Meta để khôi phục mesh 3D toàn thân,
  gồm cả bàn tay và bàn chân, từ một ảnh và các box người. LibreYOLO bọc gói
  thượng nguồn thay vì chuyển đổi mô hình.
keywords:
  - SAM 3D Body
  - khôi phục mesh người
  - mesh cơ thể
  - MHR
  - Momentum Human Rig
  - tư thế 3D
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import SAMPLE_IMAGE

        from libreyolo.models.sam3dbody import LibreSAM3DBody


        # Họ này không được đăng ký với factory LibreYOLO(), nên được khởi tạo

        # trực tiếp. model_path=None kích hoạt bản tải Hugging Face bị giới hạn;

        # còn một chuỗi được coi là đường dẫn checkpoint cục bộ hiện có và không

        # bao giờ được tự động tải. Suy luận cần thiết bị CUDA; không có luồng
        CPU.

        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])


        meshes = result.meshes

        print(meshes.vertices.shape)    # (N, V, 3), hệ tọa độ camera, mét

        print(meshes.joints3d.shape)    # (N, J, 3)
    - label: Với detector người
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        from libreyolo.models.sam3dbody import LibreSAM3DBody


        # Không có lối tắt bằng chuỗi tên ở đây: hãy truyền một detector
        LibreYOLO

        # đã khởi tạo, một callable thông thường hoặc một instance
        PersonDetector.

        detector = LibreYOLO("LibreRFDETRn.pt")

        model = LibreSAM3DBody(None, size="d3", device="cuda")


        result = model(SAMPLE_IMAGE, person_detector=detector)
source_hash: 8edc8d7872f3f875
---

## Cài đặt

```bash
pip install libreyolo
```

Lệnh này chỉ cung cấp adapter của LibreYOLO. Bản thân SAM 3D Body không được
đóng gói kèm vì giấy phép của nó không cho phép mã riêng của LibreYOLO tạo sản
phẩm phái sinh: hãy clone repo thượng nguồn và tự cài các dependency, sau đó trỏ
LibreYOLO đến bản clone.

```bash
git clone https://github.com/facebookresearch/sam-3d-body
pip install roma einops yacs omegaconf braceexpand pytorch-lightning timm
```

```python
from libreyolo.models.sam3dbody import LibreSAM3DBody

model = LibreSAM3DBody(
    None,
    size="d3",
    sam_3d_body_path="/path/to/sam-3d-body",
    device="cuda",
)
```

hoặc đặt biến môi trường `SAM_3D_BODY_PATH` thay vì truyền `sam_3d_body_path`
trong mọi lời gọi. Người dùng không khởi tạo họ này sẽ không kích hoạt thao tác
import và không gặp SAM License. Họ này không được nối vào factory `LibreYOLO()`
hay lệnh CLI `libreyolo predict`; `LibreSAM3DBody` là điểm vào duy nhất.

## Dự đoán

<code-tabs name="predict" />

Bản tải checkpoint bị giới hạn truy cập: bạn phải chấp nhận giấy phép của Meta
trên trang mô hình Hugging Face và xác thực bằng `hf auth login` trước khi lần
tải đầu tiên thành công. Bản thân suy luận luôn cần thiết bị CUDA: estimator
thượng nguồn chuyển batch sang GPU mà không kiểm tra, vì vậy máy chỉ có CPU sẽ
phát sinh lỗi thay vì dùng luồng dự phòng. `result.meshes` là payload `Meshes`,
căn theo hàng với `result.boxes` (mỗi người được phát hiện tương ứng một hàng):
`vertices` và `joints3d` dùng đơn vị mét và đã gồm phép tịnh tiến camera ước lượng,
`joints2d` dùng pixel trên ảnh gốc, còn phép quay tuân theo quy ước của MHR, tức
góc Euler thay vì trục-góc. Xem [dự đoán](/docs/predict) để biết về nguồn,
streaming và xử lý kết quả.

## Các biến thể

Có hai backbone phía sau cùng mô hình cơ thể MHR: `d3` dùng bộ mã hóa DINOv3
ViT-H/16+, còn `h` dùng bộ mã hóa ViT-H nguyên bản.

## Xuất

<export-matrix />

Chưa triển khai xuất mesh cơ thể: LibreYOLO chưa định nghĩa giao diện đồ thị đã
xuất cho tác vụ mesh, bao gồm cách biểu diễn bố cục tham số MHR bên ngoài PyTorch.

## Checkpoint

Mọi tệp trọng số đã công bố cho họ này.

<checkpoint-table />

## Giấy phép

<provenance-box>

Mô hình cơ thể được các checkpoint vận hành, MHR (Momentum Human Rig), là bản
phát hành riêng của Meta theo Apache-2.0. LibreYOLO lấy tài nguyên TorchScript
của mô hình từ bản phát hành công khai riêng của MHR trong runtime và lưu cục
bộ; LibreYOLO không tạo bản sao tệp đó, và tệp tuân theo các điều khoản
Apache-2.0 riêng chứ không phải SAM License.

</provenance-box>

## Trích dẫn

<citation-block />


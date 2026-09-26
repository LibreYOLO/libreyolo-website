---
title: Chưng cất tri thức
seo_title: Chưng cất tri thức trong LibreYOLO
description: >-
  Huấn luyện detector nhỏ theo teacher lớn hơn hoặc backbone DINOv2 đóng băng:
  các loss MGD, CWD và feature-MSE, điểm lấy đặc trưng và phạm vi hỗ trợ theo
  family.
lead: >-
  Chưng cất thêm một thành phần loss thứ hai để kéo các feature map trung gian
  của student về phía feature map của teacher đóng băng. LibreYOLO lấy đặc trưng
  bằng forward hook, vì vậy head và loss riêng của teacher không bao giờ tham
  gia.
keywords:
  - chưng cất tri thức
  - masked generative distillation
  - channel-wise distillation
  - chưng cất đặc trưng
  - teacher dinov2
  - huấn luyện teacher student
  - mgd loss
  - cwd loss
last_verified: 1.5.0
snippets:
  detector:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Checkpoint lớn hơn trong cùng family giám sát checkpoint nhỏ.
        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            distill_model="LibreYOLO9c.pt",
            distill_loss_type="mgd",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 distill_model=LibreYOLO9c.pt distill_loss_type=mgd
  foundation:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # ViT self-supervised đóng băng giám sát một stage backbone.
        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            distill_model="dinov2",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 distill_model=dinov2
  tuned:
    - label: Điều chỉnh loss
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            distill_model="LibreYOLO9c.pt",
            distill_loss_type="cwd",
            dis=1.0,           # trọng số chưng cất toàn cục
            distill_tau=1.0,   # nhiệt độ softmax CWD
        )
source_hash: 7210031328f6826f
---

## Chưng cất từ checkpoint lớn hơn

Đặt `distill_model` sẽ bật chưng cất. Giá trị này là checkpoint teacher, được
nạp qua cùng factory như mọi mô hình khác.

<code-tabs name="detector" />

Teacher chạy forward trong `no_grad` và trong autocast khi bật AMP, vì vậy mô
hình đóng băng không phải trả chi phí tính toán full precision ở mọi bước.
Forward hook thu thập feature map tại các điểm lấy đặc trưng đã đặt tên, loss so
sánh chúng với feature map của student, rồi kết quả được cộng vào loss huấn
luyện và báo cáo dưới dạng thành phần có tên `distill`.

## Chưng cất từ foundation backbone đóng băng

Thay vào đó, một ViT self-supervised có thể giám sát một stage backbone duy nhất
của student. Đặc trưng của teacher đến từ feature extractor riêng thay vì hook,
và loss xử lý sự khác biệt giữa patch grid và stride tích chập.

<code-tabs name="foundation" />

`distill_model` nhận dạng `dinov2`, tức DINOv2-base, cùng `dinov2_vits14`,
`dinov2_vitb14`, `dinov2_vitl14`, `dinov2-small`, `dinov2-base`,
`dinov2-large` và mọi hub id thô bắt đầu bằng `facebook/dinov2`. Mọi giá trị
khác được xử lý như đường dẫn checkpoint teacher.

Đường dẫn này dùng `feat_mse` bất kể `distill_loss_type` và cần cài
`transformers`. Teacher được nạp với key trọng số bị thiếu sẽ dừng thay vì chưng
cất theo một backbone ngẫu nhiên một phần.

## Các family được hỗ trợ

Hỗ trợ chưng cất là một phương thức trên mô hình student, và có hai phương thức.

`get_distill_config()` cung cấp các điểm lấy đặc trưng đa tỷ lệ mà detector
teacher giám sát. YOLOv9, YOLOX và RF-DETR triển khai phương thức này.

`get_backbone_distill_config()` cung cấp một stage backbone duy nhất mà
foundation teacher giám sát. YOLOv9 triển khai phương thức này và là family duy
nhất làm vậy.

Mọi trường hợp khác đều phát sinh lỗi thay vì huấn luyện mà không có loss:

```text
LibreDFINE does not implement get_distill_config(). Distillation is not yet
supported for the 'dfine' family.
```

```text
Foundation-model distillation into the 'yolox' family is not supported yet
(no get_backbone_distill_config()).
```

## Điểm lấy đặc trưng

Các điểm lấy đặc trưng được cố định theo từng family và từng vai trò, vì vậy
teacher và student không cần có cùng kiến trúc mà chỉ cần stride đặc trưng khớp
nhau.

| Family | Vai trò | Điểm lấy đặc trưng | Stride |
|---|---|---|---|
| YOLOv9 | teacher hoặc student | `neck.elan_up2`, `neck.elan_down1`, `neck.elan_down2` | 8, 16, 32 |
| YOLOv9 | foundation student | `backbone.elan3` | 16 |
| YOLOX | teacher hoặc student | `backbone.C3_p3`, `backbone.C3_n3`, `backbone.C3_n4` | 8, 16, 32 |
| RF-DETR | teacher hoặc student | `model.backbone.0.projector.stages.0` | được thăm dò khi thiết lập |

Stride không khớp sẽ phát sinh lỗi trước khi huấn luyện bắt đầu:

```text
Teacher and student must have matching strides. Teacher: [8, 16, 32],
Student: [16]
```

Kiểm tra đó được bỏ qua cho foundation teacher vì mục đích của chúng chính là
xử lý các grid khác nhau.

## Ba loại loss

`distill_loss_type` chọn feature loss cho detector teacher. Foundation teacher
luôn dùng `feat_mse`.

`mgd`, masked generative distillation, che một phần vị trí không gian của
student và huấn luyện generator gồm hai convolution nhỏ để tái tạo toàn bộ
feature map của teacher từ phần còn lại. `distill_mask_ratio` đặt tỷ lệ bị che,
mặc định là 0.65.

`cwd`, channel-wise distillation, chuyển activation không gian của từng kênh
thành phân phối xác suất và tối thiểu hóa KL divergence theo từng kênh.
`distill_tau` là nhiệt độ softmax, mặc định là 1.0.

`feat_mse` căn chỉnh kênh của student với teacher bằng convolution 1x1, đổi kích
thước grid của teacher theo kiểu bilinear cho bằng grid của student, rồi tính
sai số bình phương trung bình. `distill_normalize=True` chuẩn hóa L2 cả hai
feature map theo chiều kênh trước, khiến phép khớp chỉ dựa trên góc và bất biến
với tỷ lệ. Giá trị mặc định là `False`.

`dis` là trọng số toàn cục áp dụng phía trên. Khi không đặt, mỗi loss dùng giá
trị mặc định đã công bố riêng: 2e-5 cho MGD, 1.0 cho CWD và 1.0 cho feature MSE.
Chúng chênh nhau năm bậc độ lớn, vì vậy trọng số đã điều chỉnh cho một loại loss
không có ý nghĩa với loại khác.

<code-tabs name="tuned" />

`distill_mask_ratio`, `distill_tau` và `distill_normalize` không có flag CLI.
Chúng là đối số Python hoặc key YAML trong `cfg=`. Toàn bộ cơ chế chưng cất của
RF-DETR cũng chỉ dùng được qua Python vì ánh xạ đối số CLI của family này không
mang các key chưng cất.

## Adapter, checkpoint và multi-GPU

Mỗi loss dựng các module nhỏ có thể huấn luyện nằm ngoài student: các adapter
kênh 1x1 và generator của MGD. Chúng có nhóm tham số optimizer riêng ở learning
rate hiệu dụng của lượt chạy.

Các module đó được ghi vào checkpoint dưới key `distiller` và khôi phục khi tiếp
tục, vì vậy lượt chạy được tiếp tục không khởi động lại projector từ trạng thái
lạnh.

Trong DDP, các adapter nằm ngoài student được bọc, nghĩa là reducer DDP không
bao giờ thấy gradient của chúng. Trainer thực hiện all-reduce các gradient này
một cách tường minh ở mỗi bước, nên mọi rank huấn luyện cùng một adapter.

Không thể capture CUDA graph trong lượt chạy chưng cất. Truyền
`cuda_graph=True` sẽ ghi một dòng log và huấn luyện ở chế độ eager. Xem [Hiệu
năng huấn luyện](/docs/train/performance).

## Nội dung liên quan

- [Đóng băng lớp](/docs/train/layer-freezing) và [Tinh chỉnh
  LoRA](/docs/train/lora), cả hai đều có thể kết hợp với chưng cất.
- Xem [Siêu tham số](/docs/train/hyperparameters) để biết phần còn lại của
  `train()`.

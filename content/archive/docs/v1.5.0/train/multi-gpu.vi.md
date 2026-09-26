---
title: Huấn luyện multi-GPU
seo_title: Huấn luyện multi-GPU trong LibreYOLO
description: >-
  Huấn luyện trên nhiều GPU với device="0,1". Cách thư viện khởi tạo worker DDP,
  lý do batch là batch toàn cục, khi nào cần đặt sync_bn và cách dùng torchrun.
lead: >-
  Huấn luyện multi-GPU trong LibreYOLO dùng DistributedDataParallel của PyTorch:
  mỗi GPU có một process, mỗi process giữ một bản sao đầy đủ của mô hình và một
  shard của từng batch, với gradient được lấy trung bình trên các rank ở mỗi
  bước.
keywords:
  - huấn luyện pytorch ddp
  - huấn luyện multi gpu
  - torchrun nproc_per_node
  - distributed data parallel
  - syncbatchnorm
  - kích thước batch toàn cục
  - nccl gloo backend
  - multi gpu trên windows
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Bắt buộc phải có guard __main__: mỗi worker được tạo sẽ import lại

        # module này, và nếu không có guard, nó sẽ khởi chạy lại huấn luyện đệ
        quy.

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(
                data="my-dataset.yaml",
                epochs=100,
                batch=32,     # batch toàn cục: 16 ảnh mỗi GPU trên hai GPU
                device="0,1",
            )
  torchrun:
    - label: train.py
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(data="my-dataset.yaml", epochs=100, batch=32)
    - label: Khởi chạy
      language: bash
      code: |
        torchrun --nproc_per_node=2 train.py
  syncbn:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreRTDETRr18.pt")
            model.train(
                data="my-dataset.yaml",
                batch=32,
                device="0,1",
                sync_bn=True,
            )
  autobatch:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            # Chỉ thăm dò một lần trên GPU 0, rồi điều chỉnh thành bội số của world size.
            model.train(data="my-dataset.yaml", batch=-1, device="0,1")
source_hash: 83c1563d68068cd0
---

## Chạy trên hai GPU

Truyền một danh sách thiết bị. Không có gì khác thay đổi.

<code-tabs name="train" />

Khi nhận nhiều hơn một thiết bị và không có môi trường torchrun, `train()` của
mô hình lưu trọng số vào tệp tạm thời, phân giải autobatch nếu được yêu cầu, rồi
tạo một process worker cho mỗi GPU bằng `torch.multiprocessing.spawn`. Mỗi
worker import lại class mô hình, dựng lại mô hình từ trọng số đã lưu và chạy
đường dẫn một thiết bị thông thường, vì các biến môi trường torchrun được đặt
khi nhìn từ bên trong worker đã tạo. Checkpoint tốt nhất của rank 0 được nạp lại
vào instance mô hình của bên gọi khi lượt chạy kết thúc.

`device` chấp nhận `"0,1"`, `[0, 1]`, `0`, `"cuda:0"`, `"cpu"`, `"mps"` và
`"auto"`. Chỉ danh sách gồm nhiều hơn một index CUDA mới kích hoạt việc tạo
process.

## Bắt buộc phải có guard `__main__`

Các worker được tạo sẽ import lại module nguồn. Nếu không có guard
`if __name__ == "__main__":`, thao tác import đó thực thi lại lời gọi huấn luyện
và mỗi worker lại tạo các worker riêng. Thư viện phát hiện trường hợp này và
phát sinh lỗi thay vì cho phép đệ quy:

```text
spawn_ddp_train() was called from inside a spawned subprocess. This usually
means your script calls model.train(device=...) at the top level without a
'if __name__ == "__main__":' guard.
```

Mọi thứ được truyền vào worker đều được pickle, vì vậy `callbacks=` phải có thể
pickle. Class ở cấp module hoạt động; closure hoặc lambda thì không, và lỗi sẽ
nêu rõ điều đó rồi chỉ tới các logger tích hợp làm phương án thay thế.

## batch là batch toàn cục

`batch` là số ảnh trong mỗi bước optimizer trên toàn bộ GPU. Dataloader của mỗi
rank được dựng ở `batch // world_size` với `DistributedSampler`, vì vậy
`batch=32` trên hai GPU nghĩa là 16 ảnh mỗi GPU, không phải 32.

Batch không chia hết cho world size sẽ phát sinh lỗi thay vì âm thầm huấn luyện
ở kích thước khác:

```text
batch=6 is the global batch and must be divisible by world_size=4: each rank
trains at batch // world_size, so this value would silently train at a
different global batch than requested. Use batch=4 or batch=8.
```

Gradient được chính DDP lấy trung bình, vì vậy loss được truyền qua mà không đổi
tỷ lệ. Nhân loss thêm với world size sẽ làm learning rate hiệu dụng tăng xấp xỉ
theo số GPU.

## Autobatch trong DDP

`batch=-1` hoạt động và trả về batch toàn cục chia hết cho world size.

<code-tabs name="autobatch" />

Trên đường dẫn spawn, phép thăm dò chạy trong process cha trên thiết bị đầu tiên
trước khi có worker nào, vì vậy mọi worker nhận một số nguyên cụ thể và không
cần phối hợp giữa các process. Trong torchrun, rank 0 thăm dò rồi broadcast kết
quả dưới dạng một long tensor duy nhất.

Phép thăm dò đo dung lượng của một GPU và nhân với world size. Khi đặt `nbs`,
batch toàn cục bị giới hạn ở `nbs` và được làm tròn xuống thành bội số của world
size, vì vậy thêm GPU sẽ giảm số bước tích lũy thay vì thu nhỏ batch trên mỗi
GPU. Cơ chế của chính phép thăm dò được trình bày tại [Siêu tham
số](/docs/train/hyperparameters).

## SyncBatchNorm

Trong DDP, các lớp BatchNorm của mỗi rank chỉ thấy shard riêng. Với
`batch // world_size`, shard đó có thể nhỏ đến mức running statistic làm suy
giảm mô hình hội tụ so với lượt chạy một GPU.

`sync_bn=True` chuyển mọi BatchNorm thành SyncBatchNorm để statistic được tính
trên batch toàn cục. Việc chuyển đổi chỉ xảy ra khi chế độ phân tán đang hoạt
động, vì vậy flag này không ảnh hưởng tới lượt chạy một GPU ở cả hai giá trị.

Tùy chọn này đã được bật mặc định cho các family tích chập dùng nhiều BatchNorm:
YOLOX, YOLOv7, YOLOv9 và các biến thể, YOLO-NAS, PicoDet, RTMDet và FOMO. Mọi
family khác mặc định tắt. Khi mô hình chứa BatchNorm, `sync_bn` bị tắt và batch
trên mỗi rank nhỏ hơn 16, trainer sẽ cảnh báo.

<code-tabs name="syncbn" />

Không có flag CLI cho `sync_bn`. Đây là đối số Python.

## Khởi chạy bằng torchrun

torchrun cũng hoạt động và là lựa chọn phù hợp khi cluster scheduler đã quản lý
việc khởi chạy process. Viết script cho một thiết bị rồi để torchrun đặt môi
trường rank.

<code-tabs name="torchrun" />

Không kết hợp hai cách. Khi có môi trường torchrun, `device="0,1"` không tạo
process; trainer dùng `cuda:LOCAL_RANK` và torchrun quản lý số lượng process.

## Hành vi theo rank

Rank 0 sở hữu mọi side effect. Nó phân giải thư mục lượt chạy và broadcast tên
đã phân giải để mọi rank thống nhất, ghi checkpoint và artifact, đồng thời kích
hoạt callback và logger của người dùng. Các rank khác huấn luyện và đóng góp
gradient.

Mỗi rank tạo seed khác nhau cho dataloader và RNG augmentation, bắt nguồn từ
`seed` đã cấu hình, vì vậy các rank không lấy những augmentation giống hệt nhau.

## Nền tảng và backend

Backend được chọn tự động: NCCL khi cả CUDA lẫn NCCL đều khả dụng, nếu không thì
dùng Gloo. NCCL không được build trên Windows, vì vậy các lượt chạy Windows dùng
Gloo mà không cần cấu hình. Process group được khởi tạo với timeout ba giờ.

## Nội dung không chạy trong DDP

- Capture CUDA graph. `cuda_graph=True` ghi một dòng log và huấn luyện ở chế độ
  eager. Xem [Hiệu năng huấn luyện](/docs/train/performance).
- Profiler huấn luyện. `profile=True` bị bỏ qua kèm cảnh báo.

Không phải mọi family đều hỗ trợ spawn tự động. Có 24 family hỗ trợ, bao phủ các
family phát hiện, phân loại, ngữ nghĩa và phục hồi có thể huấn luyện. Khi nhận
thiết bị multi-GPU, family không hỗ trợ sẽ phát sinh lỗi nêu tên API mô hình và
lệnh torchrun thay vì âm thầm huấn luyện trên một GPU.

## Nội dung liên quan

- Xem [Siêu tham số](/docs/train/hyperparameters) để biết về `batch`, `nbs` và
  tiếp tục huấn luyện.
- Xem [Logger thí nghiệm](/docs/train/loggers) để biết ràng buộc về khả năng
  pickle của callback.
- Xem [GPU đám mây](/docs/train/cloud-gpus) để thuê máy multi-GPU.

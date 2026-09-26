---
title: Đóng băng lớp
seo_title: Đóng băng lớp trong khi huấn luyện với LibreYOLO
description: >-
  Đóng băng một phần mô hình để transfer learning: số nguyên chỉ số lượng nhóm
  đóng băng của family, danh sách index tường minh, hoặc selector theo tên
  module và tham số.
lead: >-
  Đóng băng giữ cố định các trọng số đã chọn trong khi phần còn lại của mô hình
  được huấn luyện. Selector trỏ tới các nhóm đóng băng có thứ tự hoặc tên module
  riêng của một family, không phải số lớp thô từ graph YAML.
keywords:
  - đóng băng lớp mô hình
  - transfer learning
  - đóng băng backbone
  - frozen batchnorm
  - nhóm đóng băng
  - chỉ fine tune head
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # 10 nhóm đầu tiên là toàn bộ backbone YOLOv9.
        model.train(data="my-dataset.yaml", epochs=50, freeze=10)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=50 freeze=10
    - label: Theo tên
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, freeze="backbone")
    - label: Nhiều selector
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", freeze=["backbone", "neck"])
  groups:
    - label: Liệt kê các nhóm đóng băng của family theo thứ tự
      language: python
      code: |
        from libreyolo import LibreYOLO9
        from libreyolo.models.yolo9.trainer import YOLO9Trainer

        model = LibreYOLO9("LibreYOLO9s.pt", size="s")
        trainer = YOLO9Trainer(model=model.model, wrapper_model=model, size="s")

        for index, (name, _module) in enumerate(trainer.get_freeze_groups()):
            print(index, name)
source_hash: 9f1e7551af6b16fe
---

## Đóng băng một phần

`freeze` là tùy chọn và mặc định không đóng băng gì.

<code-tabs name="train" />

Việc đóng băng diễn ra sau khi dựng mô hình và sau mọi lần dựng lại head cho số
lượng lớp đối tượng mới, nhưng trước khi tạo optimizer, vì vậy optimizer chỉ
nhận các tham số có thể huấn luyện.

## Selector có thể là gì

| Giá trị | Ý nghĩa |
|---|---|
| `None`, `False`, `""`, `"none"` | Huấn luyện mọi tham số |
| `10` hoặc `"10"` | Đóng băng mười nhóm đóng băng đầu tiên của family |
| `[0, 3, 7]` | Đóng băng các nhóm có index bắt đầu từ 0 này |
| `"backbone"` | Đóng băng nhóm, module hoặc tiền tố tham số khớp |
| `["backbone", "neck"]` | Đóng băng từng selector được liệt kê |
| `["backbone", 3]` | Có thể dùng danh sách hỗn hợp |

Một chuỗi được parse trước khi diễn giải, vì vậy CLI và cấu hình YAML chấp nhận
cùng dạng như Python. `freeze="[0, 3, 'head']"` được parse thành danh sách
literal, `freeze="backbone,neck"` được tách tại dấu phẩy, còn chuỗi chỉ chứa số
thập phân trở thành một số lượng.

`freeze=True` bị từ chối vì không rõ nghĩa.

Selector theo tên khớp với tên nhóm đóng băng, tên module hoặc tiền tố tên tham
số, và có thể dùng các ký tự glob `*`, `?` và `[`. Tiền tố `model.` được xử lý
linh hoạt, vì vậy cả `backbone` lẫn `model.backbone` đều khớp với cách viết mà
family sử dụng nội bộ.

## Nhóm do từng family định nghĩa

Một số nguyên trỏ tới danh sách nhóm đóng băng có thứ tự riêng của family, không
phải vị trí trong một graph dùng chung. Các family của LibreYOLO không cùng là
một mô hình tuần tự được lập index bằng YAML, vì vậy số lớp thô sẽ mang ý nghĩa
khác nhau trên từng family.

YOLOv9 sắp xếp các nhóm từ phía đầu vào: mười stage backbone, sau đó sáu stage
neck rồi đến head. Vì thế `freeze=10` chính xác là backbone. `backbone`, `neck`
và `head` là các selector tên ổn định đặt trên cấu trúc đó.

Các nhóm của RF-DETR là `backbone.encoder`, `backbone.projector`, `decoder`,
`queries`, `transformer.encoder_output` và `head`. Ở đây tên là lựa chọn tốt hơn
vì các thành phần transformer không ánh xạ sang số lượng lớp. `backbone` khớp cả
hai nhóm backbone theo tiền tố.

Các family không định nghĩa nhóm có ngữ nghĩa sẽ dùng giá trị mặc định thận
trọng: từng child trực tiếp của mô hình sở hữu ít nhất một tham số, theo thứ tự
khai báo. Đây thường là danh sách ngắn, vì vậy một số nguyên lớn sẽ không tìm
được đủ nhóm:

```text
freeze index 10 is out of range for 3 available freeze groups.
```

Để xem danh sách thực thay vì đoán:

<code-tabs name="groups" />

## Lỗi luôn được báo rõ

Mọi cách cấu hình sai đều phát sinh lỗi thay vì huấn luyện một thứ bạn không yêu
cầu.

Selector không khớp gì sẽ phát sinh lỗi và nêu tên các selector bị trượt:

```text
freeze selector(s) matched no parameters: 'backbon'
```

Thiết lập đóng băng khiến không còn gì có thể huấn luyện sẽ phát sinh lỗi cả ở
thời điểm đóng băng lẫn khi dựng optimizer:

```text
freeze would leave no trainable parameters. Use a smaller freeze value or
target a narrower module.
```

Đó chính là điều `freeze="all"` gây ra vì `all` khớp với mọi tham số.

Khi đóng băng thành công, một dòng sẽ ghi lại những gì đã xảy ra:

```text
Layer freezing: selectors=[10], tensors=124, params=2103776, trainable=1863456/3967232
```

## BatchNorm bị đóng băng sẽ ngừng cập nhật

Một tham số bị đóng băng vẫn nằm trong module mà running statistic có thể tiếp
tục thay đổi. Mọi module kiểu BatchNorm có tham số thuộc tập bị đóng băng sẽ
được chuyển sang eval mode, và trainer áp dụng lại điều đó sau lời gọi
`model.train()` của mỗi epoch, nhờ vậy statistic giữ nguyên trong suốt lượt chạy.

Đây là hành vi mặc định và là điều khiến việc đóng băng backbone thực sự đóng
băng nó.

## Kết hợp với LoRA

`freeze` và `lora=True` hoạt động cùng nhau. Trên RF-DETR, DEIM và ConvNeXt, các
tham số adapter vẫn có thể huấn luyện ngay cả khi nhóm cha bị đóng băng. Đây là
tổ hợp cần dùng: backbone đóng băng với adapter học trên đó. Xem [tinh chỉnh
LoRA](/docs/train/lora).

## Phạm vi

Đây là cơ chế đóng băng tĩnh được quyết định khi khởi động. Interface không có
thao tác bỏ đóng băng theo lịch hoặc đóng băng dần.

## Nội dung liên quan

- Xem [Siêu tham số](/docs/train/hyperparameters) để biết phần còn lại của
  `train()`.
- Xem [Chưng cất](/docs/train/distillation) để biết cách khác nhằm đưa kiến thức
  của mô hình lớn vào một lượt huấn luyện.

---
title: Changelog
seo_title: "LibreYOLO changelog: recent releases"
description: "What landed in LibreYOLO 1.6.0 and what 1.5.0 contained: three new tasks, six-task GTR, video embeddings, Qwen3-VL fine-tuning and opt-in torch.compile."
lead: "A summary of the repository changelog: what 1.6.0 adds, and what 1.5.0 and 1.4.0 shipped before it."
keywords: [libreyolo changelog, libreyolo release notes, libreyolo 1.6.0, libreyolo 1.5.0, libreyolo 1.4.0, libreyolo new models]
last_verified: "1.6.0"
---

## 1.6.0, release preparation

These notes describe library dev at `c21e4ae8fc8f03aed0f0200d177965ee5d1e4de8`. The v1.6.0 tag and release date are pending. See [upgrading](/docs/upgrade) for migrations.

### Highlights

<!-- H01 -->
- **Three new tasks. (#845, #846, #847, #848, #850, #857, #858)**
  3D boxes, albedo and robot action chunks join the existing result contracts.

<!-- H02 -->
- **More detection, pose, semantic and classification models. (#752, #759, #750, #745, #751, #836, #868, #894)**
  PP-YOLOE, TinyFormer, DEKR, YOLO-NAS oriented boxes, PP-LiteSeg, U-Net, ConvNeXt V2 and six-task GTR expand the model choices.

<!-- H03 -->
- **Video representations. (#746, #747, #748, #832)**
  Perception Encoder, V-JEPA 2 and LeVJEPA add finite-clip embeddings with explicit export and training limits.

<!-- H04 -->
- **Image restoration and matting. (#799, #800, #803)**
  BEN2, QuickSRNet, DDColor, HVI-CIDNet, LaMa and ViTMatte add specialized image-to-image workflows.

<!-- H05 -->
- **Language, grounding and VLM training. (#802, #804, #791, #764, #788, #861, #767)**
  LibreGround and LibreLLM add sibling APIs; LibreVLM gains model choices and Qwen3-VL detection fine-tuning.

<!-- H06 -->
- **Validation explains more of its results. (#890, #902, #883, #778)**
  Error images, per-image box counts, classification macro metrics and detection confidence estimates expose distinct evaluation views.

<!-- H07 -->
- **Training control. (#812, #875, #900, #782, #819, #905)**
  Class filtering, custom fitness, opt-in training helpers, opt-in `torch.compile` and guardless local DDP extend existing training paths.

<!-- H08 -->
- **Data and deployment workflows. (#789, #780, #865, #893, #897)**
  Hub and FiftyOne integration, event histograms, TFLite INT8 I/O and expanded result plotting are available.

<!-- H09 -->
- **Release QA fixes. (#907)**
  GPU runs and API, CLI and docs audits against 1.5.0 fixed 22 user-visible bugs; resume, NumPy input channel order, batched inputs, validation scheduling and CLI exit codes change behavior (see Changed and Breaking changes).

### Added

<!-- A31 -->
- **Histogram-based calibration algorithms. (#774)**
  Quantization accepts `algorithm="mse"` and `algorithm="entropy"` in Python and `libreyolo quantize`. They choose activation clipping ranges by quantization-error or KL-divergence sweeps; `algorithm="auto"` continues to use min/max calibration.

<!-- M01 -->
- **Three new task contracts. (#845, #850, #857)**
  `detect3d`, `albedo` and `act` extend the canonical task list from 17 to 20. New exported payloads `Boxes3D`, `AlbedoMap` and `Actions` integrate with `Results`.

<!-- M02 -->
- **PP-YOLOE detection. (#752)**
  `LibrePPYOLOE` adds `s`, `m`, `l` and `x` at 640, detection-dataset training, two-stage ATSS/task-aligned assignment and class-head rebuilding. ONNX, TorchScript, TensorRT and OpenVINO export paths are available. Pretrained files are downloaded from the source CDN and SHA-256 checked; they are not mirrored.

<!-- M03 -->
- **TinyFormer detection. (#759, #903)**
  `LibreTinyFormer` adds trainable `s`, `m`, `l`, `x` and `xl` sizes, with parsing and conversion for `-visdrone` and `-obj2coco` variants. ONNX and OpenVINO are the recorded export paths; rectangular inference is rejected. Hosted weights retain Apache-2.0 plus DINOv3 terms.

<!-- M04 -->
- **DEKR bottom-up pose. (#750)**
  DEKR with the HRNet-W32 backbone (`size="w32"`) adds inference-only COCO-17 multi-person pose without a separate person detector, using the released `no_dc` graph at 640. `Results.boxes` are fitted to confident decoded joints. ONNX, TorchScript, TensorRT and OpenVINO export paths emit heatmaps and offsets for decoding. Weights are fetched from the source CDN.

<!-- M05 -->
- **YOLO-NAS gains oriented-box training and inference. (#745, #907)**
  The oriented-box models come in sizes `s`, `m` and `l`, use a 1024 canvas and 18 DOTA2 classes, and return the existing `Results.obb` payload. Training uses rotated assignment/losses and `metrics/mAP50-95(OBB)` for checkpoint selection; `load_detect_weights_for_obb()` initializes from detection weights. ONNX, TorchScript, TensorRT and OpenVINO paths are available. Augmentation is flips and HSV. Pretrained weights remain non-commercial and non-redistributable, downloaded from the source CDN. `libreyolo train` uses the OBB recipe unless options are set (#907, d6b5d27b).

<!-- G1 -->
- **GTR detection, segmentation, pose, oriented boxes, depth and semantic segmentation. (#894, 3e136716; #907)**
  `LibreGTR` adds 22 hosted weights: `LibreGTR{s,m,l,x}.pt` for COCO detection, `-seg`, `-pose` (17 COCO keypoints), `-depth` and `-sem` (19 Cityscapes classes) in the same four sizes, and `LibreGTR{s,x}-obb.pt` for the 15 DOTA v1.0 classes. Inputs are square, a multiple of 32 and at least 160 pixels, default 640; OBB uses a fixed 1024 canvas, and semantic segmentation slides 1024-pixel windows over a 1024×2048 canvas. Postprocessing is NMS-free. Depth returns relative inverse depth, so `1 / depth_map` gives upstream's meters. CPU and MPS run a portable PyTorch recurrence; CUDA uses `flash-linear-attention` when it is installed separately. All six tasks train, in FP32 by default. Detection defaults to 30 epochs, batch 4, AdamW and the flat-cosine schedule (other optimizers and schedulers raise), with upstream Mosaic and MixUp in the first 6 epochs. `lora=True` covers detection, segmentation, pose and OBB. Pose training requires a single-class 17-keypoint dataset at 640. ONNX and TorchScript export fixed-shape FP32 graphs; other formats are rejected. Resume continues the checkpoint's run with the call's device, `exist_ok=False` opens a new run, and `pretrained=` accepts only `True` or `False` (#907, a30759d4, 1777942b, 30decab6, 59019f1b). Code is MIT with retained Apache-2.0 notices; weights are MIT, as the upstream weight repository declares.

<!-- M06 -->
- **PP-LiteSeg semantic segmentation. (#751, #907)**
  `LibrePPLiteSeg{t50,b50,t75,b75}-sem.pt` adds trainable Cityscapes segmentation. Evaluation uses 512×1024 for `t50`/`b50` and 768×1536 for `t75`/`b75`; default training crops are 512×1024 and 768×768 respectively. The recipe includes auxiliary heads, Dice/cross-entropy/edge loss, polynomial learning-rate decay and EMA. ONNX, TorchScript, TensorRT and OpenVINO support the native rectangles. `resume=True` resumes the loaded checkpoint and keeps writing into its run, a path resumes that file, and `libreyolo train` keeps the size-aware recipe unless options are set (#907, ad52e7f5, 878464cd).

<!-- M07 -->
- **U-Net semantic segmentation. (#836)**
  `LibreUNets-sem.pt` is the trainable same-padded UNet-S5-D16 graph with an FCN head, converted from mmsegmentation: 19 Cityscapes classes, 1024×2048 evaluation and 512×1024 training crops, with cross-entropy plus auxiliary loss. Export is unsupported. Weights are Apache-2.0, as mmsegmentation declares.

<!-- M08 -->
- **ConvNeXt V2 classification. (#868)**
  Eight sizes, `atto`, `femto`, `pico`, `n`, `t`, `b`, `l` and `h`, add 224-pixel classification, raw-checkpoint conversion, supervised fine-tuning/resume and ONNX/TorchScript export. The constructor defaults to `size="atto"`. Code is MIT; official ImageNet-1K weights retain CC-BY-NC-4.0.

<!-- M09 -->
- **Perception Encoder image, text and video embeddings. (#746, #892)**
  `LibrePE` adds `t16`, `s16`, `b16`, `l14` and `g14`: zero-shot classification through `set_classes()`, image/text embeddings and one embedding for a finite video. `clip_frames=8` controls uniform frame sampling; frame embeddings are averaged and L2-normalized. Canonical weights use `-cls`; select `task="embed"` for embeddings. Training is unsupported. ONNX/TorchScript include image and fixed-frame video graphs; exported video graphs require direct runtime input. Weights are Apache-2.0.

<!-- M10 -->
- **V-JEPA 2 video embeddings and attentive probes. (#747, #748, #892)**
  Four encoders, `l256`, `h256`, `g256` and `g384`, and four released classification probes support finite-video clips and still images. `embed_tokens()` returns `(B, T', H', W', D)`; the public embedding is the normalized mean of final tokens. Frozen-encoder probe training validates video manifests and selects checkpoints by top-1 accuracy. ONNX/TorchScript embedding graphs take direct 5D clip input; generic exported video preprocessing is unsupported. Full encoder fine-tuning and pretraining are unsupported. The two giant encoder artifacts are Apache-2.0; the other six artifacts are MIT.

<!-- M11 -->
- **LeVJEPA clip and patch embeddings. (#832)**
  `LibreLeVJEPAl-embed.pt` adds inference-only 16-frame ViT-L/16, normalized 1024-dimensional CLS embeddings and patch tokens through `embed_tokens()`. Finite videos use a centered window sampled at approximately 7.5 FPS. TorchScript export requires direct input at batch 1, 16 frames and 224×224. Weights retain CC-BY-NC-4.0.

<!-- M12 -->
- **BEN2 background removal. (#799)**
  `LibreBEN2b-matte.pt` adds fixed-1024 background removal, batched native prediction, matte validation and transparent cutout saving. ONNX/TorchScript exports use fixed resolution and batch 1. Weights are MIT.

<!-- M13 -->
- **QuickSRNet 2× super-resolution. (#800)**
  `LibreQuickSRNetm2-restore.pt` adds native-resolution Medium 2× upscaling, paired PSNR/SSIM validation, dynamic-spatial ONNX and fixed-canvas TorchScript export. Weights are BSD-3-Clause.

<!-- M14 -->
- **Colorization, low-light enhancement, inpainting and guided matting. (#803)**
  `LibreDDColor{t,l}-restore.pt` colorizes images; `LibreHVICIDNett-restore.pt` enhances low light with `gamma=1.0`, `saturation=1.0` and `intensity=1.0`; `LibreLaMab-restore.pt` requires `mask=` for inpainting; `LibreViTMattes-matte.pt` requires `trimap=` for matting. Python and predict CLI accept the single-image guides. All four support paired validation and reject training/export. LaMa runs an embedded ONNX graph.

<!-- M15 -->
- **Four 3D detection APIs. (#845, #846, #847, #848, #851)**
  `LibreWildDet3D`, `Libre3DMOOD`, `LibreFCOS3D` and `LibreDetAny3D` return `Results.boxes3d`, with dedicated `wilddet3d`, `3dmood`, `fcos3d` and `detany3d` CLI commands. FCOS3D is native; the other three require separately installed upstream runtimes. WildDet3D, 3D-MOOD and FCOS3D require camera `intrinsics`; DetAny3D estimates them. The payload stores center, dimensions and a `wxyz` quaternion in camera coordinates, plus confidence/class fields and intrinsics. Plotting projects cuboids. Training, validation, tracking and export are unsupported for all four.

<!-- M16 -->
- **Marigold V2 depth, normals and albedo. (#850)**
  Nine adapters use a separately downloaded pinned base model, default `quantization="4bit"` and `seed=2025`. Four-bit inference requires CUDA; `quantization="none"` enables a CPU code path, whose full execution was not verified. `AlbedoMap` stores linear RGB and converts to sRGB for display; albedo validation reports PSNR/SSIM in linear RGB and uses PSNR as fitness. Training and export are unsupported. Adapters and base model are Apache-2.0.

<!-- M17 -->
- **LibreGround instruction-to-point API. (#802, #804)**
  `LibreGround()` defaults to ShowUI-2B, with Florence-2 base/large and Qwen3-VL 2B/4B/8B adapters. `prompt=` or `query=` sets a per-call instruction; `set_query()` persists it. Single-image multi-query calls merge at most one click per query into `Results.points`. This is inference-only. Florence-2 and ShowUI weights are MIT; Qwen3-VL weights are Apache-2.0.

<!-- M18 -->
- **Additional LibreVLM model choices. (#764, #788, #861)**
  Gemma 4 E2B/E4B adds detection, with `gemma-4` selecting E4B; Moondream 2/3 adds detection, points and native chat; North Micro Vision runs one detection query per vocabulary class; `lfm2-vl-3b` adds LFM2.5's 3B size and 0–1000 box parsing. Molmo2 4B/8B/O-7B returns points, and custom pointing templates must include `{label}`. These additions are inference-only. Moondream 3 retains BSL 1.1 restrictions, and LFM retains its Open License; the other listed snapshots are Apache-2.0.

<!-- M19 -->
- **Qwen3-VL detection fine-tuning. (#767, #834, #907)**
  `LibreVLM("qwen3-vl-2b").train(data=...)` adds LoRA by default, optional full language-model fine-tuning with the vision tower frozen, checkpoint-directory reload, callbacks/loggers and validation-loss checkpoint selection. Defaults include `epochs=10`, `batch=1`, `accumulate=8`, `workers=0`, `hflip=0.5` and gradient checkpointing. `lr0=None` resolves to `1e-4` for LoRA or `2e-5` for full language-model tuning. Runs default to `runs/vlm/train` with `exist_ok=False`. Optimizer-state resume, COCO-JSON training and detection-mAP validation are unsupported. `resume=True` on a model loaded from `<run>/weights/<best|last>` continues and writes into that run, and a resumed LoRA adapter is applied to the base weights once (#907, ce888888).

<!-- M20 -->
- **LibreVLA robot-policy API. (#857, #858)**
  `LibreVLA()` defaults to SmolVLA; ACT and Diffusion Policy provide training from scratch without a pretrained policy base. Prediction accepts camera frames and `state=`, returning an action chunk `(T, D)` with names, control rate and instruction. LeRobot-data training defaults to `epochs=5`, `batch=8`, `val_split=0.1` and `workers=0`. Offline validation reports aggregate L1/MSE, first-action L1 and per-dimension L1. Checkpoints are directories with `libreyolo_vla.json`. Diffusion keeps observation history; use `reset()` between episodes. No CLI, tracking or export is provided. ACT may download its pretrained image backbone.

<!-- M21 -->
- **LibreLLM compatible-endpoint client. (#791)**
  `LibreLLM` supports `api="responses"` by default or `api="chat.completions"`, image requests via `image=`, streaming, `async_call()`, custom `base_url` and `openai/`/`openrouter/` provider prefixes. It returns native SDK responses. The constructor's default model is `gpt-5.6-luna`; no local weights are loaded.

<!-- M24 -->
- **RF-DETR UI detector weights. (#897)**
  `LibreRFDETRm-ui.pt` adds a class-agnostic screenshot detector with the single class `object`, routed to `LibreYOLO/LibreRFDETRm-ui`. Only the medium detection variant is accepted. The weight declaration is MIT.

<!-- A01 -->
- **Validation error images. (#890, #902)**
  `val(visualize=True)` draws class-aware box TP/FP/FN for detection/segmentation and label versus top-1 for ImageFolder classification. Matching uses IoU `0.5` and confidence `max(0.25, conf)`. Images go to `visualize/errors/` or `visualize/correct/` under the run directory. Defaults are `visualize=False`, `show_labels=True`, `show_conf=True`; all are exposed on the val CLI. Unsupported tasks and the separate V-JEPA 2 clip validator reject visualization.

<!-- A02 -->
- **Per-image validation box metrics. (#902)**
  Detection and segmentation `val()` results remain dictionary-compatible and expose `.box.image_metrics`, keyed by image filename, with `precision`, `recall`, `f1`, `tp`, `fp` and `fn`. The first occurrence keeps its basename; subsequent duplicates use full paths. Counts use the visualization matching rule even when `visualize=False`; segmentation counts boxes. Zero denominators return `0.0`. These records are not gathered across distributed ranks.

<!-- A03 -->
- **Image classification macro metrics. (#883)**
  ImageFolder validation adds `metrics/precision`, `metrics/recall` and `metrics/f1`, averaging over classes present in validation targets, with zero precision for unpredicted classes. Top-1/top-5 keys remain, and default fitness remains top-1. CLI text and JSON include the new values. Targets outside the classifier head's range raise a mismatch error.

<!-- A04 -->
- **F1-optimal detection confidence thresholds. (#778, #907)**
  Detection validation adds `metrics/best_conf` and `metrics/best_conf_f1`, computed at IoU `0.50` with micro F1 over classes; per-class thresholds, keyed by class name, are on `results.box.best_conf_per_class`. Equal-score detections are grouped, and F1 ties choose the higher threshold. When no threshold reaches F1 > 0 the threshold and F1 are `0.0`, so the returned metrics stay a flat dict of finite numbers (#907, 9831fa94). Segmentation's separate metrics implementation does not expose these values.

<!-- A05 -->
- **Configurable validation sample plots. (#880)**
  `plot_samples=8` sets the retained sample-image budget on validation, training configuration and both CLI commands. `0` disables sample images and `-1` retains all. It does not limit evaluated images or the separate `visualize=True` output.

<!-- P19 -->
- **`val()` without `data=` reuses the training dataset. (#907, 625cfbe0, a725680d, 4bee7b7a)**
  Native `model.val()` with no `data`, `data_dir` or `keypoints_json` validates on the dataset path saved in the loaded checkpoint's training configuration, or on the dataset of a `train()` that finished in the same session. Released weights carry none and raise `ValueError` asking for `data=`.

<!-- P20 -->
- **`val(project=, name=, exist_ok=)`. (#907, 6595f5e8)**
  Native and exported-backend `val()` write to `project/name` (defaults `runs/val` and `exp`), incremented unless `exist_ok=True`; combining them with `save_dir` raises. Without either, output keeps the timestamped `runs/val/` directory.

<!-- A06 -->
- **Two-channel event-histogram detection. (#865)**
  YOLO9 and RF-DETR detection accept prepared `(H, W, 2)` `.npy` arrays containing finite, nonnegative positive/negative counts. Dataset `input_profile` declares `format="event_histogram"`, `layout="HWC"`, `polarity="positive_negative"`, `encoding="counts"`, positive `scale` and positive integer `window_us`. Preprocessing clips to `scale`, divides by it and resizes without RGB normalization. RGB-weight initialization sets both input channels to `1.5 * mean(R,G,B)`; checkpoints preserve profile and initialization metadata. Native train/val/predict and FP32 ONNX without embedded NMS are supported. This path requires fixed-batch, single-device training and excludes live event decoding, video, TTA, tiling, LoRA, distillation and quantization.

<!-- A07 -->
- **Custom tracker instances. (#869, #907)**
  `track(tracker=instance)` accepts the exported `libreyolo.tracking.Tracker` protocol: `reset()` and `update(results, image=None)`. Each run resets the instance once, unless `persist=True` continues it from the previous `persist=True` call on the same model and stream (#907, 48542ae9), and passes the original RGB PIL frame. Returned `track_id` must be a one-dimensional integer array/tensor aligned with boxes and on the matching backend/device. Custom instances use `track_conf=0.25` by default; configure them directly instead of passing `tracker_config` or tracker kwargs.

<!-- A08 -->
- **Tracking across image sequences. (#825, #907)**
  `track()` accepts individual images, filename-sorted folders, lists, tuples and lazy image iterators as consecutive frames. New `fps=30.0` and `color_format="auto"` control timing and the channel order of NumPy frames: `"auto"` reads them as BGR, like `cv2` frames, and `"rgb"` as RGB (#907, d80e1627). `vid_stride` reduces the output rate to `fps / vid_stride`; ByteTrack and BoT-SORT use that retained rate unless explicitly configured.

<!-- P21 -->
- **`track(..., persist=True)` keeps the tracker across calls. (#907, 48542ae9)**
  A call with `persist=True` continues the tracker and track IDs of the previous `persist=True` call on the same model, for per-frame loops such as `model.track(frame, persist=True)`. A different tracker or configuration, or another video file or directory, starts a fresh tracker; the default `persist=False` always does.

<!-- A09 -->
- **Hub checkpoint loading and publishing. (#789, #907)**
  `LibreYOLO("owner/repo")` or `LibreYOLO("hf://owner/repo@revision/path/model.pt")` loads schema-tagged checkpoints through the Hub cache; existing local paths take precedence. `model.push_to_hub("owner/repo", private=False)` publishes `model.pt` and a model card. `HuggingFaceHubLogger` or `loggers="hf:owner/repo"` uploads `weights/best.pt`, falling back to `last.pt`, at training end; the logger defaults to `private=True`. Ambiguous repositories require an explicit filename; a sole `.pt` takes precedence over `.safetensors` candidates. A bare `owner/name` whose last segment has a file extension (`weights/model.pt`, `ckpts/best.pth.tar`) or starts with `librefacerec-` is treated as a local path (#907, 5c8a9acb).

<!-- A10 -->
- **FiftyOne model application and dataset exchange. (#780)**
  `libreyolo.integrations.fiftyone` exports `apply_model`, `to_fiftyone_model`, `to_fiftyone` and `from_fiftyone`. Model application supports detect, segment, pose, OBB and classify. Dataset exchange covers detection/segmentation YOLO layouts and native COCO annotations, with curated views exportable to training YAML. `apply_model` defaults to `label_field="predictions"`, `conf=0.25`, `iou=0.45`, `max_det=300`, `mask_format="mask"` and `skip_failures=True`.

<!-- A12 -->
- **TFLite INT8 inputs and outputs for YOLOX and YOLO9 detection. (#893)**
  `export(format="tflite", int8=True, data=..., fraction=1.0, batch=1)` creates separate normalized-box and score outputs with independent quantization scales. The backend uses sidecar `output_layout` metadata to reconstruct predictions. Missing `data` falls back to `coco8.yaml` with a warning. Requires `onnx2tf[tensorflow]`; `half=True`, `dynamic=True`, other INT8 families/tasks and batch sizes other than 1 are rejected. Some internal operators may remain floating point.

<!-- A13 -->
- **Expanded `Results.plot()` rendering. (#897, #907)**
  Detection, masks, pose, OBB, classification top-5, points, OCR, semantic/panoptic results and other image-overlay payloads use a shared renderer. Overlay output is contiguous `HxWx3` uint8 BGR; `pil=True` requests PIL. Dense maps (depth, normal, edge and albedo), 3D cuboids and action chunks return a PIL image by default. New controls include `img`, `conf=True`, `labels=True`, `boxes=True`, `masks=True`, `probs=True`, `line_width=None`, `pil=None`, `show=False`, `save=False` and `filename=None`. `orig_img` is an `HxWx3` uint8 BGR array: in-memory and URL sources keep their decoded pixels, and local image or video files are read back from `path` on first access (#907, f4da8e89).

<!-- T01 -->
- **Class-subset and single-class detection training. (#812, #875, #838, #907)**
  `train(classes=[...])` filters supervision while preserving original class IDs and dataset `nc`/`names`; `single_cls=True` collapses retained labels to class 0, named `object`. Defaults are `classes=None`, `single_cls=False`. Both options cover detection with YOLO9 (including YOLO9-E2E and YOLO9-P2), RF-DETR, RT-DETR, RT-DETRv2, RT-DETRv4, D-FINE, DEIM, DEIMv2, EdgeCrafter, GTR, TinyFormer and YOLO-NAS; resuming or validating a subset-trained checkpoint inherits its saved settings, while a new run validates with its own `classes`/`single_cls` (#907, 3ce36e24). CLI train/val expose class subsets, and `libreyolo val --single-cls` evaluates the same detectors with classes merged (#907, 9d77b56b). The public OBB dataset loader can collapse classes, but model OBB training rejects `single_cls`.

<!-- T02 -->
- **Minimum epoch sample count. (#777)**
  `min_samples=0` adds an opt-in floor for shared detection loaders, including RF-DETR detection. A shorter dataset is sampled with replacement; DDP rounds draws to equal rank lengths and reshuffles by epoch. Specialized loaders and custom sampler paths retain their own behavior.

<!-- T03 -->
- **Class-balanced detection sampling. (#782)**
  `class_balanced=False` enables repeat-factor sampling when set to true on shared detection loaders, using per-image class occurrence, exponent `0.5` and the median positive class frequency as threshold. It combines with `min_samples` and DDP. Empty images retain weight 1; single-class collapse makes balancing uniform. Trainers bypassing the shared sampler reject enabled balancing.

<!-- T04 -->
- **Averaging selected training checkpoints. (#782, #900)**
  `average_best=0` defaults off; `average_best=N` writes `weights/average.pt` from up to N best-scoring snapshots. Floating tensors are averaged uniformly, and integer/boolean buffers come from the best snapshot. The pool persists for supported ordinary resume, and final validation is attempted separately.

<!-- T05 -->
- **An export check before epoch one. (#782)**
  `export_check=False` defaults off; enabling it writes `export_check.onnx` and fails the run if export fails. Native/ONNX raw outputs are compared at `rtol=1e-3`, `atol=1e-4` only when tensor layouts and dtypes match; otherwise the skipped comparison is logged. ONNX Runtime is required.

<!-- T06 -->
- **Final BatchNorm recalibration. (#782)**
  `precise_bn=0` defaults off; `precise_bn=N` consumes up to N training-loader images before final validation, preserves frozen BatchNorm and refreshes a historical best checkpoint when needed. Under DDP the budget is per participating rank before moment reduction. Models without BatchNorm are unchanged.

<!-- T07 -->
- **Custom training fitness callbacks. (#900)**
  One callback may implement the exported `TrainFitnessCallback` protocol's `fitness(metrics)` and return a finite, higher-is-better scalar for `best.pt`, patience and checkpoint averaging. Raw metrics remain; scores use `fitness/custom` and checkpoints record `fitness_source="callback"`. Custom-fitness runs cannot resume, and an ordinary resumed run cannot add a scorer: load weights and start `resume=False`. The hook is Python-only and rejected by VLM/VLA trainers.

<!-- T08 -->
- **Classification loss weighting. (#841, #842, #868)**
  `cls_pw=0.0` accepts `[0,1]`, computing `n_c ** (-cls_pw)` normalized to arithmetic mean 1. Alternatively, `class_weights=True` uses `N / (C*n_c)`; its default is false, and it cannot combine with `cls_pw>0`. Supported families are ResNet, ConvNeXt, ConvNeXt V2, MobileNetV4, EfficientNetV2 and DINOv2. Weighting settings must match on resume.

<!-- T09 -->
- **Classification crop and augmentation controls. (#877, #879, #881)**
  `scale=(0.5,1.0)` accepts a lower-bound float or explicit crop-area pair; `crop_pct=None` preserves the family's evaluation ratio. Python and CLI accept the new crop controls; CLI adds `auto_augment`, `erasing`, `cutmix`, `fliplr` and `flipud`. Classification CLI `mixup` now routes to batch mixing, default `0.0`, rather than detection `mixup_prob`. `flip_prob=0.5` and `flipud=0.0` control flips. A `crop_pct` override affects train/val evaluation; export retains native family preprocessing.

<!-- T10 -->
- **Guardless automatic local DDP. (#819, #907)**
  `model.train(device=[0,1])` and `device="0,1"` use coordinator-managed ranks without replaying an ordinary unguarded training script's top-level code. Guarded scripts and explicit `torchrun` remain supported. Callback/logger objects requiring the documented standard-pickle fallback still need a main guard. This includes the default DataLoader `workers>0`: rank worker processes do not re-run the script (#907, b034110d).

<!-- G7 -->
- **Opt-in `torch.compile` training. (#905, 0ce0d51c, bfad4596, 7b39ffe5; #907, 5e63bbe5)**
  `train(compile=...)` accepts `False` (default), `True` or `"default"`, `"reduce-overhead"`, `"max-autotune"` and `"max-autotune-no-cudagraphs"`; other values raise `ValueError`, and `libreyolo train compile=` exits `config_type_error` for them. The network forward and backward are compiled at the boundary CUDA-graph capture uses, including YOLO9's default PGI branch and RF-DETR detection; the loss, optimizer, EMA, validation, checkpoints and export stay eager, and checkpoints load without compilation. Only single-GPU CUDA runs compile. CPU, MPS, distributed and distillation runs, tasks without a capture boundary (including RF-DETR segmentation, pose, OBB and classification) and compiler failures train eager after a warning. `"reduce-overhead"`, `"max-autotune"` or `cuda_graph=True` replay CUDA graphs, except under gradient accumulation, which RF-DETR's defaults (`batch=4`, `nbs=16`) use. With RF-DETR's default multi-scale training, dynamic-shape compilation failed in release GPU testing and the run continued eager; the warning suggests `multi_scale=False`, which compiled (#907, 5e63bbe5). On an RTX 4090, RF-DETR nano at batch 4 trained 1.44x faster and YOLO9-t at batch 16 did not speed up; compilation takes several minutes.

<!-- T11 -->
- **RF-DETR multi-class pose. (#888)**
  Dataset `kpt_names`, keyed by class index or name, retains the first `len(names)` keypoint rows per class; `[]` marks a box-only class. Head resizing and checkpoints retain `num_keypoints_per_class`, and predictions pad keypoints to `kpt_shape`, zeroing box-only classes. Multiclass datasets require `names`, and at least one class must have keypoints. Default keypoint-mAP fitness does not score box-only classes.

<!-- T12 -->
- **Rectangular semantic training datasets. (#751)**
  `SemanticDataset` accepts `(height, width)`, adds `rescale_crop` sampling with ignore padding and supports a family-provided `photometric` transform. The shared `PolyLRScheduler` supports the semantic recipes.

### Changed

<!-- P1 -->
- **Output change: NumPy image inputs are read as BGR by default. (#907, d80e1627)**
  `predict()`, `track()`, exported backends, `LibreEnsemble` and `ImageLoader.load()` treat 3-channel NumPy arrays as BGR when `color_format` is `"auto"` (the default) or `"bgr"`, matching `cv2.imread()` and OpenCV frames; `color_format="rgb"` reads RGB arrays. In 1.5.0 `"auto"` meant RGB, so `model(cv2.imread(path))` saw swapped channels. PIL images, tensors, files, bytes and video files are unchanged.

<!-- P2 -->
- **Output change: batched arrays and tensors return one `Results` per image. (#907, 1df8cec2)**
  A 4-D NumPy array (NHWC or NCHW) or 4-D tensor is split into its images and returns a list with one `Results` each, also for a batch of one. 1.5.0 used only the first image and returned a single `Results`. `ImageLoader.load()` raises `ValueError` for a batch larger than one.

<!-- P12 -->
- **Output change: RF-DETR fine-tunes keep pretrained class logits when dataset class names match the checkpoint. (#907, 020f8eaf)**
  When every dataset class name matches a checkpoint class, the classification head is rebuilt from the matching rows; the released 91-column COCO head maps class names to COCO category-id columns. 1.5.0 truncated the head to `nc + 1` columns, which shifted every label on COCO-named data (coco8 fine-tune mAP 0.053 before, 0.642 after). Other datasets keep the truncate/tile resize, with a warning that the rows do not correspond to the dataset classes.

<!-- P13 -->
- **Output change: ONNX INT8 export keeps each family's float layers. (#907, e9743d71)**
  Unless `nodes_to_exclude` is given, the layers `model.quantize()` keeps in float stay float in ONNX INT8 export: YOLO9 `head.` and `backbone.conv0.`, RF-DETR class, box, angle, keypoint and segmentation heads, embeddings and reference points, and the listed BiRefNet/FeyNobg modules. YOLO9 INT8 scores no longer saturate at 0.5: coco8 mAP rises from 0.322 to 0.471 (FP32 0.527). Re-export INT8 models to pick this up.

<!-- P14 -->
- **Output change: exported MiDaS and Depth Anything V2 resize inputs bicubically. (#907, 6a33162d)**
  Their exported backends resize the `[0, 1]` float image with bicubic interpolation, as native prediction does, instead of bilinear on uint8, so their depth maps change (MiDaS ONNX had differed about 9% from `.pt`). Other depth families are unchanged.

<!-- DC22 -->
- **Output change: EdgeCrafter pose keypoints match upstream. (#894)**
  At inference the keypoint position embedding also reaches the attention value, residual, gate input and the previous decoder layer's refinement features, as in upstream EdgeCrafter; 1.5.0 added it to queries and keys only. Released checkpoints reproduce the upstream decoder to float rounding (logits had moved by up to 2.1), and coco8-pose keypoint mAP50-95 rises from 0.186 to 0.813 (S) and from 0.438 to 0.818 (M). EC pose predictions and exports made with 1.5.0 differ from 1.6.0. Training keeps the query/key-only form.

<!-- G12 -->
- **Output change: YOLO-format detection and segmentation validation no longer counts a duplicate detection as a true positive. (#906, d1948c44)**
  The first prediction of each evaluation had id 0, which COCO evaluation reads as "unmatched", so the ground truth it matched stayed free and a lower-scoring duplicate of that object also counted as a true positive. Prediction ids now start at 1, as in pycocotools. At most one ground truth per evaluation was affected, so box and mask mAP and AR can fall slightly, most on small validation sets. COCO-JSON datasets and `faster_coco_eval=True` were not affected.

<!-- P5 -->
- **The final training epoch always validates, and `val=False` turns validation off. (#907, b9faf668, 6f038fcf, 0cc2598d)**
  With validation on, training validates every `eval_interval` epochs and after the final epoch, so runs shorter than `eval_interval` (10 for YOLO9, YOLOX, YOLOv7, D-FINE, DEIM, DEIMv2, RT-DETRv4, TinyFormer, EC detect/segment, YOLO-NAS detect, PP-YOLOE, PicoDet, RTMDet and Dome-DETR; 5 for EC pose) report metrics and write `best.pt`, and `best.pt` can come from the final epoch. Python `train(val=False)` sets `eval_interval=0` and skips all validation, including final plots and precise-BN metrics; 1.5.0 warned about `val` as an unknown key and validated anyway. `libreyolo train val=false` is honored for RF-DETR too.

<!-- P3 -->
- **Resume continues the saved run: its arguments, directory and checkpoint. (#907, 606144f4, a5816bc4)**
  For YOLO9 (incl. E2E and P2), YOLOX, YOLOv7, D-FINE, DEIM, DEIMv2, RT-DETR, RT-DETRv2, RT-DETRv4, EC, YOLO-NAS, PP-YOLOE, PicoDet, RTMDet, Dome-DETR, TinyFormer, FOMO, ConvNeXt, ConvNeXt V2, ResNet, MobileNetV4, EfficientNetV2, U-Net and NAFNet, `train(resume=True)` resumes the checkpoint the model was loaded from and `train(resume="path/to/last.pt")` resumes that file. Saved arguments such as `data`, `epochs`, `imgsz`, `batch`, `lr0` and `amp` are restored unless passed; `device` follows the call. A `<run>/weights/*.pt` checkpoint keeps writing into `<run>`, and `exist_ok=False` opens a new numbered run beside it. Released weights raise `ValueError` ("holds no training state") instead of `KeyError: 'epoch'`, and a finished run raises `ValueError` instead of training on. The CLI forwards only the options you set when resuming, so `libreyolo train model=<run>/weights/last.pt resume=true` works for YOLO9, and a missing checkpoint exits `checkpoint_not_found`. In 1.5.0 resume took the call's defaults (for YOLO9, a new 300-epoch run in a new directory) and resumed the loaded weights even when a path was given.

<!-- P4 -->
- **RF-DETR resume restores the run's settings. (#907, 46bbff45, d19bb608, 59a347eb, 9f70b876)**
  `LibreRFDETR.train(resume=True|path)` takes the dataset, epochs, batch, lr0 and the other saved settings from the checkpoint unless passed, and continues that run's directory; the CLI does the same and forwards only the options you set. 1.5.0 resumed with the signature defaults (100 epochs, batch 4, lr 1e-4), and the CLI opened a new run directory.

<!-- P6 -->
- **Validation during training writes to `<run>/val`. (#907, 654a17b3)**
  Every family's in-training validation writes `config.yaml`, plots and JSON inside the run folder. 1.5.0 created a `runs/val/<tag>_<time>/` directory in the working directory for each validation when plots were off.

<!-- P7 -->
- **Python `train()` honors `mosaic`, `fliplr` and detection `mixup`. (#907, baa54963, 073aeaab, 7a15208f)**
  `mosaic` sets `mosaic_prob`, `fliplr` sets `flip_prob`, and on non-classification models `mixup` sets `mixup_prob`, as the CLI does; on classification `mixup` stays batch MixUp. 1.5.0 ignored `mosaic` and `fliplr` with an unknown-key warning and routed detection `mixup` to the unused classification field. A call's value overrides the same setting from `cfg=`; an alias and its field with different values in one source raise `ValueError`.

<!-- P9 -->
- **`export(quantize=16|8|32)` selects precision. (#907, f8e12af1, 7c6d0119)**
  `16` exports FP16 (`half=True`), `8` INT8 (`int8=True`) and `32` FP32, in Python and with `libreyolo export quantize=`; other values or a conflicting `half`/`int8` raise. 1.5.0 accepted `quantize=16` silently and wrote FP32.

<!-- P10 -->
- **Unused export options are warned about. (#907, f8e12af1)**
  `export()` warns `Unknown <format> export arguments (ignored)` for keyword options the format's exporter does not take; 1.5.0 accepted them silently.

<!-- P11 -->
- **`track(conf=...)` sets the detection threshold. (#907, 220d9a72)**
  Detections below `conf` no longer reach the tracker, as with `predict(conf=)`. Without `conf` the detector keeps running at the tracker's low threshold. 1.5.0 warned about `conf` as an unknown tracker key and ignored it.

<!-- P15 -->
- **CLI YOLO-NAS pose training uses the pose recipe. (#907, d6b5d27b)**
  `libreyolo train` for YOLO-NAS and PP-LiteSeg forwards only the options you set, so YOLO-NAS pose uses its Python recipe (lr0 2e-3, weight decay 1e-6, 10 warmup epochs, 1000 epochs, AMP). 1.5.0 applied the detection recipe to pose (lr0 5e-4, weight decay 1e-5, 1 warmup epoch, 300 epochs, FP32).

<!-- P16 -->
- **CLI exit codes follow the failure type. (#907, b5b9d406, d2c56c84, 7861400f)**
  Training failures from bad values or arguments exit 2 with `config_type_error`, `config_unknown_key` or `config_unsupported`; CUDA out-of-memory exits `cuda_oom` and missing device operators `device_not_available`; other failures stay `io_error` (exit 1). An unavailable CUDA/MPS device exits `device_not_available` instead of `model_load_failed`, an `imgsz` the model cannot run exits `invalid_imgsz`, a missing resume checkpoint exits `checkpoint_not_found` instead of `data_not_found`, and `libreyolo metadata` on an unreadable file exits `model_load_failed` instead of a traceback. 1.5.0 reported every training failure as `io_error`.

<!-- P17 -->
- **The CLI rejects out-of-range values. (#907, ebd26372)**
  `conf` and `iou` outside [0, 1], and `max_det`, `batch` or `epochs` below 1, exit `config_range_error` (exit 2) in predict, val, export and train; `batch=-1` still requests AutoBatch.

<!-- P18 -->
- **Unpublished weights fail fast, and `libreyolo models` lists only routable names. (#907, 9546891f, acef0fb1, 926d740a)**
  A weight URL answering 401, 403, 404 or 410 raises `WeightsNotPublishedError` ("No weights are published at ...") without retries. Names known to be unpublished, such as CLIP `l14`, Depth Anything V2 `g`, DINOv2 heads, EoMT `s`/`b` seg/sem and YOLOv1 `t`, explain why instead of downloading. `libreyolo models` drops names without a download route, lists known-unpublished names separately, and adds `cli_command`, `unpublished_names` and `python_class` to `--json`. It does not probe the network, so names whose repository is missing are still listed.

<!-- P8 -->
- **Unknown training keys suggest the close key. (#907, 6f2938e1)**
  `train(epoch=1)` warns `Unknown training config keys (ignored): ['epoch'] (did you mean 'epochs'?)` and still trains the default epochs; nothing is auto-corrected. Keys that are ecosystem arguments LibreYOLO lacks, such as `lrf`, get no suggestion.

<!-- DC23 -->
- **Auto-converted upstream downloads use a new cache name. (#752)**
  An upstream checkpoint downloaded under its canonical name converts to `<canonical>-converted.pt` instead of `<canonical>-<canonical>.pt`. Existing 1.5.0 caches convert once more on first load, and the old file stays until deleted. Other sources keep `<source stem>-<canonical>.pt`.

<!-- C01 -->
- **Expected optional-runtime fallbacks are quieter. (#811, #867)**
  Video output logs an H.264 fallback at INFO and caches failed probes by `(codec, width, height)` only after a later codec successfully opens the output. Missing xFormers in the depth DINOv2 layers is logged at DEBUG.

<!-- C02 -->
- **Package metadata and CLI text are updated. (#898, #773)**
  PyPI metadata gains keywords, classifiers and website, documentation, issue and changelog links. CLI help and status strings use plain punctuation. Export-support `since` metadata is restamped as release-line metadata; it is not a guarantee that a capability existed in v1.5.0.

<!-- C03 -->
- **Sponsorship links use Open Collective. (#855, #889)**
  The funding configuration, README badges and sponsorship policy point to the LibreYOLO collective.

<!-- C04 -->
- **Depth results carry an explicit encoding. (#850)**
  `DepthMap(..., encoding="inverse_depth")` keeps the existing default and adds `"depth"` and `"log_depth"`. Depth validation interprets the encoding before affine alignment; relative predictions do not acquire a metric-scale guarantee.

<!-- C05 -->
- **RF-DETR preprocessing changes detection results. (#897)**
  Detection, segmentation and OBB prediction/validation use floating-point OpenCV bilinear resize without antialiasing instead of PIL downscaling. Exported backends use the same preprocessing. Boxes, scores and validation metrics can change for the same checkpoint and input; pose retains its separate antialiased resize.

<!-- C06 -->
- **Classifier evaluation uses family-specific preprocessing. (#892)**
  Classification validation and INT8 calibration use the model's evaluation transform. Export metadata adds `norm_mean`, `norm_std` and `resize_mode`; exported ViT, CLIP and SigLIP2 validation uses their own normalization instead of ImageNet defaults. Legacy exports fall back to family values, so their validation scores can also change.

<!-- C07 -->
- **Saved classification predictions include the top five labels. (#897)**
  `predict(save=True)` and exported-backend saving use the shared result renderer, including tracking IDs. Matte saving still produces an RGBA cutout, while `plot()` composites the cutout over a checkerboard.

<!-- C08 -->
- **Classification augmentation can close for the final epochs. (#881)**
  `no_aug_epochs` disables `auto_augment`, `erasing`, `mixup` and `cutmix` in the final training tail while retaining crop and flips. Classification family configurations default that tail to `0`. Existing classifier helper imports remain re-exported from `data.classify_dataset` after the implementation moves to `data.augment.classify`.

<!-- C09 -->
- **Four detector recipes enable AMP by default. (#754, #907)**
  D-FINE, DEIM, RT-DETRv4 and YOLO-NAS detection now use `amp=True`; `amp_dtype="float16"` is unchanged. Dome-DETR, PP-YOLOE and YOLO-NAS OBB retain FP32 defaults. Resume restores a run's saved `amp` (#907, 606144f4), so a 1.5.0 FP32 run resumes in FP32.

<!-- C10 -->
- **YOLO9 training recipe and checkpoint geometry. (#796, #907)**
  New stock detection fine-tunes use a training-only PGI branch with `aux_weight=0.25`; `aux_weight=0` trains the main head only (Python, or `libreyolo train aux_weight=0` for YOLO9; #907, 3ce36e24). `max_labels` rises from 100 to 300 and SGD momentum warms from `0.8` to `0.937` over three epochs, for YOLO9, YOLO9-E2E and YOLO9-P2. Measured on YOLO9-s (coco128, batch 16, RTX 3090), peak training VRAM rises from 5.47 GB to 9.05 GB and training time by 37% compared with 1.5.0, so batch sizes that fit in 1.5.0 can run out of memory. Old single-head checkpoints resume as single-head. Unmarked checkpoints keep `letterbox_pad="topleft"`; new official conversions stamp `"center"`, and `letterbox_pad=None` inherits the checkpoint setting. Exports record `letterbox_pad`, and all exported backends, backend validation, INT8 calibration and DeepStream sidecars (`symmetric-padding=1` for center) use it; artifacts without the key stay top-left (#907, 0380a2d9, 9cc6e46b). Inference/export use the main head.

<!-- C11 -->
- **RF-DETR and DINOv2 use incremented training directories. (#834, #907)**
  Their Python `output_dir` default changes from `"runs/train"` to `None`, resolving to `runs/train/rfdetr_exp` and `runs/train/dinov2_exp`, with `exist_ok=False`. DINOv2's CLI default name is `dinov2_exp`. `resume=True` on a model loaded from `<run>/weights/*.pt`, or `resume="<run>/weights/last.pt"`, continues writing into that run in Python and the CLI; `exist_ok=False` opens a new numbered run beside it (#907, 3ce36e24, d19bb608, 30decab6, 59a347eb).

<!-- C12 -->
- **Mosaic and MixUp prefer annotated partners. (#776)**
  Mosaic partners are drawn with bounded retries in the shared YOLOX-style mosaic dataset (YOLOX, RTMDet, PicoDet, FOMO) and the YOLO9-style one (YOLO9 and its E2E/P2 variants, YOLOv7, RT-DETR); YOLO9-style MixUp does too. This applies where a family's recipe enables mosaic or MixUp. Up to 20 candidates are drawn, retaining the final draw if none have annotations. This changes the augmentation distribution for datasets containing empty images.

<!-- C13 -->
- **QAT disables incompatible training state. (#794, #782)**
  Quantized-model setup disables EMA and SyncBatchNorm and sets `average_best=0`, logging the changes. Ordinary floating-point training retains its requested settings.

<!-- C14 -->
- **Dataset-mutation hooks reject incompatible persistent workers. (#775, #881)**
  Training raises when active `close_mosaic`/`set_epoch` hooks cannot reach persistent multi-worker dataset copies. The default nonpersistent loader path is unaffected.

<!-- C15 -->
- **CUDA MSDA fallback offers an installation hint. (#793, #790)**
  Eager CUDA calls without an accepted accelerated provider emit one hint for `libreyolo[hub-kernels]`. `ms_deform_attn_available(value=None)` can inspect the actual tensor. `LIBREYOLO_HUB_KERNELS=0` disables Hub and its hint; Triton has a separate opt-out.

### Breaking changes

<!-- P1 (breaking) -->
- **RGB NumPy arrays passed without `color_format` are read as BGR. (#907)**
  Code that passes `np.asarray(pil_image)` or other RGB arrays gets swapped channels.
  Migration: Pass `color_format="rgb"` for RGB arrays, or pass the PIL image; `cv2.imread()` output needs no change.

<!-- P2 (breaking) -->
- **4-D array and tensor inputs return a list. (#907)**
  `model(batch)` returns one `Results` per image in a list, also for a batch of one.
  Migration: Index the list (`results[0]`) or pass a 3-D image.

<!-- P3 (breaking) -->
- **Resume restores the run instead of starting from the call's defaults. (#907)**
  Resumed runs take their saved arguments and keep writing into the checkpoint's run directory; resuming released weights or a finished run raises.
  Migration: Pass the arguments you want to change (for example a larger `epochs=` to extend a finished run), pass `exist_ok=False` for a new run directory, and train without `resume` when starting from released weights.

<!-- B01 -->
- **Environments pinning ONNX Runtime below 1.18 no longer satisfy the ONNX extra. (#803)**
  The extra's lower bound changes from 1.16 to 1.18.
  Migration: Upgrade to `onnxruntime>=1.18.0` before installing `libreyolo[onnx]`.

<!-- B02 -->
- **SAM 3D Body accepts only the reviewed checkpoint/configuration asset set. (#807)**
  An explicit checkpoint path must identify `model.ckpt` beside its matching `model_config.yaml` and `LICENSE`, or the containing directory, with the pinned bytes. Arbitrary renamed or modified assets are rejected. Extra files outside the accepted snapshot inventories are rejected.
  Migration: Use automatic acquisition after installing `libreyolo[hf]` and obtaining access to the gated model, or provide the reviewed snapshot directory and pinned MHR asset on a trusted local filesystem.

<!-- B03 -->
- **Preprocessing corrections can invalidate saved validation baselines. (#897, #892)**
  RF-DETR non-pose resizing and exported classifier normalization change without a legacy-preprocessing compatibility flag.
  Migration: Re-run validation and recheck deployed confidence thresholds before comparing metrics or outputs with v1.5.0.

<!-- B04 -->
- **Default mixed precision changes four detector training recipes. (#754)**
  D-FINE, DEIM, RT-DETRv4 and YOLO-NAS detection use FP16 autocast by default.
  Migration: Pass `amp=False` to retain FP32 training.

<!-- B05 -->
- **New stock YOLO9 fine-tunes use different training defaults. (#796, #907)**
  PGI, the ground-truth cap and momentum warmup change new-run behavior.
  Migration: Use `aux_weight=0` (also `libreyolo train aux_weight=0`), `max_labels=100` and `warmup_momentum=0.937` for the former choices; the last two also apply to YOLO9-E2E and YOLO9-P2. Lower `batch` if a 1.5.0 batch size runs out of memory. Set `letterbox_pad="topleft"` when starting from a new center-stamped conversion if the old geometry is required.

<!-- B06 -->
- **Default RF-DETR and DINOv2 artifact paths change. (#834, #907)**
  Fresh Python runs use family-named, incremented directories; resumed runs keep writing into the checkpoint's own directory (#907).
  Migration: Update artifact-path consumers, or pass `output_dir="runs/train", exist_ok=True` to retain the old location and reuse behavior.

<!-- B07 -->
- **Invalid classification augmentation combinations now raise. (#881)**
  `mixup + cutmix > 1` is rejected at setup instead of silently reducing the effective CutMix probability.
  Migration: Keep classification mixing probabilities summing to at most 1.

<!-- B08 -->
- **QAT no longer keeps requested EMA or SyncBatchNorm active. (#794)**
  The quantized-model setup overrides both options.
  Migration: Use QAT best/last checkpoints without relying on EMA or SyncBatchNorm state; averaged fake-quant checkpoints are also disabled.

<!-- B09 -->
- **Custom persistent-worker loaders may now fail at mutation hooks. (#775, #881)**
  Loaders whose worker copies cannot observe scheduled dataset changes are rejected.
  Migration: Use `persistent_workers=False` for these loaders, or rebuild workers after dataset mutation.

<!-- P16 (breaking) -->
- **CLI training errors use new codes. (#907)**
  Configuration errors exit 2 with `config_*` codes instead of `io_error` (exit 1); device, `imgsz` and missing-checkpoint failures have their own codes.
  Migration: Match the documented codes in scripts that parse `--json` errors or exit statuses.

### Deprecated

None identified relative to v1.5.0.

### Removed

None identified relative to v1.5.0.

### Fixed

<!-- F01 -->
- **YOLO text labels use the same geometry in training and validation. (#815, #827)**
  Finite boxes crossing an image border are clipped; boxes with no visible area, non-finite coordinates and malformed polygons are dropped. Training keeps segment rings aligned with accepted boxes and passes the configured class count to the parser, so out-of-range IDs are reported before target creation.

<!-- F02 -->
- **Training monitor completion counts and charts use the reported epoch. (#831)**
  `status.json` no longer adds one to the completed-epoch count, and the monitor no longer adds one to `current_epoch`, `best_epoch` or `metrics.jsonl` chart coordinates. ETA therefore includes the remaining final epoch.

<!-- F03 -->
- **RT-DETR filename matching no longer captures unrelated checkpoints. (#871, #850)**
  Local-checkpoint size detection requires a delimiter around single-character `l` and `x` codes instead of matching names such as `last.pt` or `model_xlnet.pt`. Auto-download accepts only canonical names such as `LibreRTDETRl.pt`, `LibreRTDETRx.pt` and `LibreRTDETRr50.pt`.

<!-- F04 -->
- **Custom RF-DETR pose checkpoints retain their names and device. (#884)**
  Checkpoint class names take precedence over legacy arguments instead of being replaced with `person`. Rebuilt GroupPose attention-mask buffers stay on the decoder device after a keypoint-schema change.

<!-- F05 -->
- **Skipped pose labels identify the file, line and parsing error. (#876)**
  Training and validation report malformed rows, quote at most the first five rejected lines and count the rest. Diagnostics distinguish non-numeric values from an incorrect field count and explain the expected `kpt_shape: [K, 2]` or `[K, 3]` layout.

<!-- F06 -->
- **Unavailable weight spellings fail with a useful alternative. (#773, #779)**
  Auto-download rejects task-suffixed FCN and Mask R-CNN names and names the hosted forms `LibreFCNr50.pt`, `LibreFCNr101.pt` and `LibreMaskRCNNr50.pt`. `lingbotvision-g` explains that no checkpoint is published and points to `s`, `b`, `l` or a local checkpoint.

<!-- F07 -->
- **Classifier preprocessors supply calibration arrays. (#892, #907)**
  AlexNet, VGG, ResNet, EfficientNetV2, ConvNeXt, MobileNetV4, DeiT, Swin and ViT satisfy the calibration `(CHW array, ratio)` contract, and their evaluation transforms accept the `(height, width)` size that export passes (#907, 12a2d321), so INT8 calibration no longer drops the images; non-square classification sizes raise `NotImplementedError`. DINOv2 classifier/embed calibration uses its classification pipeline instead of semantic preprocessing.

<!-- F08 -->
- **Rectangular fine-tunes restore their saved input canvas. (#901)**
  Loading restores `imgsz_h`/`imgsz_w`, and the live model adopts the training rectangle or clears it after a later square run. Default exports of square-native families that cannot use the rectangle fall back to `max(height, width)` with a warning; explicitly requested unsupported rectangles still raise. Export CLI JSON reports the actual canvas, and tiling defaults to its long side.

<!-- F09 -->
- **Rectangular auxiliary inference paths use the requested size. (#901)**
  Fixes YOLOX validation ratios, TTA letterbox reversal, YOLO9-E2E default postprocessing, `InferenceProfiler`, validation preprocessors and `capture_graph(imgsz=(h,w))`. Finalized quantized `.pt` exports retain rectangular dimensions.

<!-- F10 -->
- **DETR-line rectangular prediction handles height and width separately. (#903)**
  RF-DETR, D-FINE, DEIM and EdgeCrafter accept `imgsz=(height, width)` in their resize paths. RF-DETR still checks the task-specific patch/window grid. DEIMv2 gives an explicit `ValueError` for unsupported rectangular predict/val requests. This does not add universal rectangular training or export.

<!-- F11 -->
- **Raw checkpoints retain the correct class metadata. (#897, #745)**
  RF-DETR's one-output class head converts as `nc=1` instead of receiving an 80-class record. YOLO-NAS detect/pose conversion reads `processing_params.class_names` when ordinary argument metadata is absent.

<!-- G13 -->
- **Converted RF-DETR checkpoints with empty `names` keep their class labels. (#906, 45d62c26, caef122d)**
  Auto-conversion of a raw RF-DETR checkpoint whose top-level `names` is empty takes labels from `args.class_names` or `hyper_parameters.class_names` when their count matches the resolved class count, and a checkpoint whose metadata identifies COCO gets the COCO-80 names. 1.5.0 labeled both `class_0`, `class_1` and so on.

<!-- F12 -->
- **DINOv2 resume restores training state. (#834, #907)**
  `resume=` calls the trainer's resume path before continuing instead of only carrying the value in configuration, keeps the task's best score (`metrics/accuracy_top1` or `metrics/mIoU`) so the first resumed epoch no longer overwrites `best.pt`, continues the checkpoint's run directory, and trains on the dataset saved in the checkpoint when `data=` is omitted (#907, a0222aaa, 59a347eb, 9f70b876).

<!-- G6 -->
- **OBB validation works with DataLoader workers on macOS and Windows. (#894, a5177119)**
  The OBB validation preprocessor recursed without end when unpickled, so spawned worker processes (the default on macOS and Windows) failed with `RecursionError` whenever `workers` was above 0, as it is by default (4). In 1.5.0 this affected RF-DETR and RT-DETRv2 OBB validation.

<!-- G8 -->
- **`cuda_graph=True` with gradient accumulation sums gradients correctly. (#905, 0ce0d51c)**
  Autograd adopted a replayed graph's gradient buffers as `.grad` and the next replay overwrote them, so an accumulation window stepped with twice the last micro-batch's gradient instead of the sum. Live gradients now get their own storage before each replay when accumulating. This affected every `cuda_graph=True` run with `nbs` above `batch`, including RF-DETR's defaults (`batch=4`, `nbs=16`); runs without accumulation are unchanged.

<!-- G10 -->
- **Loading models no longer changes `torch.compile` settings for the whole process. (#905, 0ce0d51c)**
  In 1.5.0, importing `libreyolo.models`, which every model load does, set `torch._dynamo.config.automatic_dynamic_shapes=False` and `accumulated_cache_size_limit=1024` through the vendored DINOv3 backbone, affecting any model compiled in the same process.

<!-- G11 -->
- **RF-DETR no longer prints a Transformers deprecation notice. (#905, 0ce0d51c)**
  The DINOv2 backbone reads `config.return_dict` instead of `config.use_return_dict`, which recent Transformers releases report as deprecated on the first prediction.

<!-- P24 -->
- **`track(tracker="bytetrack.yaml")` works. (#907, 462f2910)**
  The `.yaml`/`.yml` spellings of the built-in tracker names select that tracker instead of raising "Unknown tracker"; a local file with that name raises, because tracker yaml files are not read.

<!-- P25 -->
- **Resuming a non-SGD run without `optimizer=` works. (#907, 3ce36e24)**
  Resume uses the checkpoint's optimizer, so an AdamW run no longer rebuilds SGD and fails with `KeyError: 'momentum'`.

<!-- P26 -->
- **DINOv2 `train(batch=...)` is honored. (#907, 43da519b)**
  Every DINOv2 run trained at batch 4 regardless of the requested batch.

<!-- P27 -->
- **An ecosystem `cls` train key no longer crashes. (#907, de7807f5)**
  `train(cls=0.5)` and `cfg=` files with the `cls` loss gain are warned about as unknown keys instead of raising `TypeError: got multiple values for argument 'cls'`.

<!-- P28 -->
- **YOLO9 rounds `imgsz` up to a multiple of 32. (#907, eda9b46d)**
  Predict, val, export and train warn and round, for example 333 to 352, instead of failing with "Sizes of tensors must match".

<!-- P29 -->
- **`predict(classes=0)` and YOLO9 `imgsz=[h, w]` lists work. (#907, e72f3396, 5a40955a)**
  Native, exported-backend and ensemble predict accept a single int class id; YOLO9 predict and export accept list sizes.

<!-- P30 -->
- **`classes=` is applied before `max_det`. (#907, 6cef64d2)**
  In predict, TTA and exported backends, `classes=[16], max_det=1` returns the best class-16 box instead of nothing.

<!-- P31 -->
- **Tiling rejects `overlap_ratio` outside [0, 1). (#907, bd3be61a)**
  Values of 1 or more raise `ValueError` instead of never finishing.

<!-- P32 -->
- **FP16 ONNX and TorchScript exports load and predict. (#907, 321a9f93, 3be89e20)**
  YOLO9 and RF-DETR `half=True` ONNX files load in ONNX Runtime, backends cast inputs to the graph dtype, and the in-memory model is restored exactly after a half export.

<!-- P33 -->
- **ONNX INT8 export works with the default `dynamic=True`. (#907, 16e26385)**
  ONNX Runtime symbolic shape inference, which failed on dynamic-batch graphs, is skipped.

<!-- P34 -->
- **`val(device=...)` moves the model. (#907, 4a071ed1)**
  As `predict(device=...)` does, instead of failing with a mixed-device error; YOLO9 rebuilds its anchor cache after a device or dtype switch, including MPS to CPU.

<!-- P35 -->
- **TorchScript exports load on Apple Silicon. (#907, 32c0c4ff)**
  `device="auto"` picks CPU instead of MPS, which cannot load the traced float64 constants.

<!-- P36 -->
- **Quantization and QAT run on Apple Silicon. (#907, 7861400f)**
  Fake-quantized models move to CPU with a warning instead of failing on MPS.

<!-- P37 -->
- **Finalized fp16-remainder checkpoints return to float32 cleanly. (#907, 32a3e933)**
  Their half-width I/O hooks are removed, so export, QAT and dequantize work.

<!-- P38 -->
- **EfficientNetV2 and EfficientDet ONNX exports match PyTorch. (#907, 31a04e9b, 0124a09f)**
  SAME padding is baked as constants when tracing, so onnxsim no longer folds it into wrong pads; re-export existing ONNX files.

<!-- P39 -->
- **YOLO9 CoreML export works and letterboxes like `.pt`. (#907, 7d0869cf, 9cc6e46b)**
  Export no longer fails with `IndexError` (empty frozen anchor grid), and the CoreML backend letterboxes with the exported pad instead of stretching; a non-square coco8 image matches `.pt` with IoU 1.0.

<!-- P40 -->
- **Fixed-shape exports name the expected size. (#907, 01ac6b3d, ef43f7a0)**
  ONNX, TensorRT, OpenVINO and CoreML exports run at another `imgsz` raise `ValueError` with both sizes; the CLI exits `invalid_imgsz` and suggests the exported size.

<!-- P41 -->
- **`val(split=...)` with an empty dataset entry names the split. (#907, e5dc0225)**
  Detection, segmentation and OBB validation raise `FileNotFoundError` instead of `TypeError: PosixPath / NoneType`.

<!-- P42 -->
- **RF-DETR `batch=-1` fits in memory. (#907, 43ad8560)**
  Losses compute paired IoU/GIoU instead of an N x N matrix (same values), and AutoBatch probes the real loss at the largest multi-scale size, so it picks a smaller batch; on a 24 GB GPU it had estimated 10.6 GiB for a step that used more than 21 GiB.

<!-- P43 -->
- **Failed training setup leaves no `status.json` in the working directory. (#907, 544b44c4)**
  Failures after setup still record `state=failed` in the run folder.

<!-- P44 -->
- **CLI `--json` output is one JSON document. (#907, e4750317, 67b1f91e, e4afc028)**
  Third-party output (pycocotools tables, torch ONNX logs) goes to stderr, and predict failures and usage errors print JSON errors with documented codes instead of tracebacks.

<!-- P45 -->
- **CLI `--key value` values are not rewritten. (#907, 4e5fc889)**
  `--name show` names the run `show` instead of `--show`.

### Performance

<!-- P01 -->
- **Training reduces matcher transfers and optimizer launches. (#761, #762, #765, #771, #772)**
  Affected RF-DETR, D-FINE/DEIM and RT-DETR matchers separate cost building from host assignment with a depth-two pipeline. Broadcast L1 costs replace `cdist(p=1)` where applicable; YOLO9 batches loss logging transfers. Eligible CUDA Adam/AdamW construction is fused; SGD and non-CUDA parameters retain stock construction.

<!-- P02 -->
- **Denoising indices and EC tensors avoid repeated work. (#772)**
  D-FINE/DEIM derive denoising positive indices from known target counts and retain tensor normalizers. EC caches eligible anchor grids and projection vectors, bypassing caches for trainable parameters.

<!-- P03 -->
- **Hub MSDA accepts half-precision inputs. (#760, #762)**
  FP16/BF16 inputs are upcast to FP32 for the kernel and outputs are restored to the value dtype, with gradients retained through the casts.

<!-- P04 -->
- **In-tree Triton MSDA for eligible CUDA inference. (#784, #787, #790)**
  The provider supports FP32/FP16/BF16 when Triton is installed, rejects gradient-requiring inputs and falls back to portable attention. Hub remains preferred; `LIBREYOLO_TRITON_MSDA=0` disables Triton. EC pose adapts its split values to the shared slot. Provider fallback, device selection, capture handling and disable-after-failure behavior are included.

<!-- G9 -->
- **Training keeps PyTorch's CPU threads within the process's CPU allowance. (#905, 0ce0d51c, 888e810a, 7b39ffe5)**
  During `train()`, unless `OMP_NUM_THREADS` is set, the PyTorch thread count is lowered to the smaller of the cgroup CPU quota and the CPU affinity count, divided among local ranks, and restored afterwards; it is never raised. This stops CPU-limited containers from throttling every step. It applies to all families on the shared trainer, not to LibreVLM or LibreVLA fine-tuning.

### Security/Licensing

<!-- S01 -->
- **Existing-family source and weight notices are completed. (#773, #779)**
  Notices add or clarify attribution and weight terms for LW-DETR, LingBot-Vision, face embedding/detection, SAM 3D Body, SenseNova-Vision and several existing classifier/detector families. FeyNobg mirror descriptions retain `fp16`/`fp8`; the withdrawn `nvfp4` claim is removed.

<!-- S02 -->
- **SAM 3D Body and MHR loading validate pinned assets. (#807)**
  SAM snapshots are keyed by immutable revisions and verify the checkpoint, configuration and license bytes. MHR acquisition checks the reviewed release archive and member identities. Integrity checks reject linked paths and compare files before and after construction; the path-based upstream constructor still requires a trusted, quiescent local filesystem.

<!-- S03 -->
- **Weight delivery preserves model-specific restrictions. (#753, #797, #862, #864, #851, #845, #846, #848)**
  MiDaS `s/l`, MoGe2 `s/l` and SAM1 base/large/huge route to LibreYOLO mirrors; MoGe2 `b` remains upstream. Six Dome-DETR mirrors are academic-research-only. Twelve EC `obj2coco` detect/segment/pose variants are opt-in and non-commercial; original COCO variants retain their recorded Apache-2.0 grant. DetAny3D weights are CC-BY-NC-4.0, WildDet3D retains the SAM license, and 3D-MOOD is Apache-2.0. These weight terms are separate from the library's MIT license. MiDaS is mirrored under the publisher's MIT grant.

### Dependencies & extras

<!-- D01 -->
- **Automatic DDP adds one core dependency. (#819)**
  `cloudpickle>=3.0.0` transports coordinator jobs. The package-wide Python requirement remains `>=3.10`.

<!-- D02 -->
- **The ONNX runtime floor rises. (#803)**
  `libreyolo[onnx]` now requires `onnxruntime>=1.18.0` instead of `>=1.16.0`, for LaMa's opset-21 graph.

<!-- D03 -->
- **New integration and language extras. (#789, #791, #780, #802, #767, #807)**
  `hf` adds `huggingface_hub>=1.0.0`; `llm` adds `openai>=1.66.0`; `fiftyone` adds `fiftyone>=1.0.0`; `ground` installs `libreyolo[vlm]`; `vlm-train` adds that VLM stack and `peft>=0.17.0`. `all` gains `hf` and `llm`. FiftyOne remains separate because its headless OpenCV dependency overlaps the core `cv2` installation.

<!-- D04 -->
- **VLA, Marigold and Molmo2 have separate dependency stacks. (#857, #858, #850, #861)**
  `vla` adds `lerobot[smolvla,diffusion,dataset]>=0.6.1` only on Python `>=3.12`. `marigold` pins `diffusers==0.38.0`, `peft==0.18.1`, `accelerate==1.13.0`, `transformers==5.4.0` and, on Linux/Windows, `bitsandbytes==0.49.2`. `molmo2` pins `transformers==4.57.1` with `einops>=0.8.0` and `accelerate>=0.26.0`. These extras are outside `all`; uv declares Molmo2 conflicts with the newer Transformers/Hub stacks, including `hf`, `vlm`, `ground`, `marigold` and `all`.

<!-- D05 -->
- **Some VLM families require newer Transformers than the common VLM extra floor. (#764, #788)**
  North Micro Vision checks for `transformers>=5.16.0`, and Gemma 4 for `>=5.10.0`, before loading; the shared `vlm` extra still declares `>=5.1.0`.

## 1.5.0, released 2026-08-09

The largest release so far: 28 new model families, four new tasks, five new
export formats, two new serving backends, and an `import libreyolo` that no
longer pulls PyTorch. 600 commits and 88 merged pull requests.

Summarized from
[`CHANGELOG.md`](https://github.com/LibreYOLO/libreyolo/blob/release/CHANGELOG.md),
which carries the full entries. Two defaults move numbers and four arguments
changed shape: [upgrading to 1.5.0](/docs/upgrade) is the short list of what
that asks of you.

### Breaking changes

No public model class or function was removed, and `__all__` grew from 101
names to 142. Four things need a code edit:

- `allow_experimental=True` is gone from every `.train()` gate. Delete the
  argument; a call that still passes it raises `TypeError`.
- The export support tier `"experimental"` is gone. `Tier` is now
  `validated`, `available` or `blocked`.
- `pretrained=False` combined with `resume` raises `ValueError` instead of
  proceeding incoherently.
- CLI `--imgsz` widened from int to str so it can carry `480x640`. Typing
  `--imgsz 640` in a shell and calling `model.predict(imgsz=640)` are both
  unaffected; only direct Python calls into the CLI command functions need a
  string.

Three changes move metrics at default settings: the COCO backend, YOLOX
BatchNorm eps on checkpoints trained before this release, and D-FINE's
per-size multi-scale recipe. All of it, with before and after code, is on
[upgrading to 1.5.0](/docs/upgrade).

### New tasks

The task vocabulary went from 13 entries to 17, with nothing removed. Each new
task ships an original-canvas result payload, visualization, dataset schema and
filename suffix.

[`edge`](/docs/tasks/edge-detection) and
[`normal`](/docs/tasks/surface-normals) are dense-prediction contracts, with
validators for edge ODS/OIS and normal angular error.

[`embed`](/docs/tasks/face-recognition) produces L2-normalized image and region
embeddings. It ships `LibreFaceEmbedder`, a `Gallery` and `FaceGallery` API,
and 1:N identification from `libreyolo predict` through `--gallery` and
`--gallery-threshold`. The task is deliberately general: identity vectors are
one use, region embeddings and re-identification are others.

[`mesh`](/docs/tasks/body-mesh) is body mesh recovery. `LibreSAM3DBody` is an
optional model gated on the `sam_3d_body` dependency, so it is not exported
from the top-level package.

### New model families

28 new user-facing classes, exported under 29 names. All are inference-only
unless stated.

Dome-DETR (`domedetr`) is the trainable addition: a tiny-object detector for
aerial, drone and remote-sensing imagery, ported from the Dome-DETR release. It
is D-FINE plus a density head (DeFE), encoder attention restricted to occupied
windows (MWAS), and a query count set by local density rather than a fixed 300
(PAQI). Sizes s, m and l at 800x800, with a maximum absolute difference of 0.0
against upstream on all six published checkpoints. Training is wired against
upstream's full objective, though the published 160-epoch schedule has not been
reproduced, so the paper's AP numbers are unverified. Its advantage narrows as
objects grow, so it sits beside D-FINE rather than replacing it. There is no
COCO checkpoint upstream, only AI-TOD-V2 with 9 classes and VisDrone with 12,
so every canonical filename carries a dataset suffix. Export raises instead of
emitting a graph, because PAQI's per-image query count makes a traced graph
valid only for the image it was traced on. Weights are not rehosted: the
upstream card claims Apache-2.0 while also restricting use to academic
research.

[LingBot-Vision](/docs/models/lingbot-vision) is the other trainable arrival, a
ViT semantic-segmentation family at 512 px pairing boundary-centric
self-supervised backbones with a 1x1 dense head. Sizes s, b and l are
published; `g` is the 1.1B teacher and has no LibreYOLO-hosted checkpoint, so
requesting it raises with the reason.

The rest are inference-only ports of earlier published work, each checked
against its pinned upstream source:

- Detection: [DETR](/docs/models/detr),
  [Deformable DETR](/docs/models/deformable-detr),
  [DINO-DETR](/docs/models/dino-detr), [LW-DETR](/docs/models/lw-detr),
  [Faster R-CNN](/docs/models/faster-rcnn), [Mask R-CNN](/docs/models/mask-rcnn),
  [FCOS](/docs/models/fcos), [RetinaNet](/docs/models/retinanet),
  [SSD300](/docs/models/ssd), [CenterNet](/docs/models/centernet) and
  [EfficientDet](/docs/models/efficientdet).
- Classification: [AlexNet](/docs/models/alexnet), [VGG](/docs/models/vgg),
  [ViT](/docs/models/vit), [DeiT](/docs/models/deit) and
  [Swin](/docs/models/swin).
- Semantic segmentation: [FCN](/docs/models/fcn) and
  [DeepLabv3](/docs/models/deeplabv3).
- Depth: [MiDaS](/docs/models/midas). Pose: [HRNet](/docs/models/hrnet),
  top-down on COCO-17 person crops.
- Surface normals: [MoGe-2](/docs/models/moge-2) at 518 px, a DINOv2 patch grid
  with letterbox preprocessing.
- Edge: [TEED](/docs/models/teed) and [DexiNed](/docs/models/dexined), native
  MIT-licensed architectures with local checkpoint converters. Their upstream
  BIPED-trained checkpoints are not bundled, mirrored or auto-downloaded,
  because that dataset's terms are non-commercial.
- Embeddings: [LibreFaceEmbedder](/docs/models/librefacerec), ONNX Runtime
  only.

[LibreMODUS](/docs/models/libremodus) adds 14B-A7B analysis-only inference for
depth, normals, edges and COCO detection from one checkpoint, plus phrase
grounding and image-conditioned `any2any()` chaining.
[LibreFeyNobg](/docs/models/feynobg) adds a matte family at 1024 px, code and
weights Apache-2.0.

[RT-DETRv2](/docs/models/rt-detr) gained
[oriented-object detection](/docs/tasks/oriented-detection) inference for the
official DOTA 1.0 checkpoints in sizes n, s, m, l and x, with
aspect-preserving preprocessing, native `Results.obb` output, validation, and
validated ONNX and TorchScript export.

### Torch-free ONNX inference

`import libreyolo` no longer pulls torch. Model and results names resolve
through a lazy `__getattr__`, and a new top-level `libreyolo.preprocess`
package provides numpy-native preprocessing for deim, deimv2, dfine, ec,
rfdetr, rtdetr, yolo9, yolonas and yolox, so an ONNX Runtime install needs no
torch at all. See [lightweight install](/docs/lightweight-install).

### Export and serving

Five new export formats: [RKNN](/docs/export/rknn) for Rockchip, with `--name`
for the target platform and `--verify` for PC-simulator versus ONNX Runtime
parity; [MNN](/docs/export/mnn); [Paddle](/docs/export/paddle);
[ExecuTorch](/docs/export/executorch); and [Core AI](/docs/export/coreai) on
macOS.

[DeepStream](/docs/export/deepstream) sidecar configs are written by
`deepstream=True` on [ONNX](/docs/export/onnx) export. It is an ONNX-only
option rather than a `format=` key, and is mutually exclusive with `nms=True`.
ExecuTorch and MNN exports now require a checkpoint sidecar,
`<program>.pte.json` and `<model>.mnn.json`.

Two new inference backends: [Triton](/docs/export/triton), with
`create_triton_config` for NVIDIA Triton Inference Server, and Paddle
Inference.

### New CLI commands

`libreyolo enroll` builds an embedding gallery from a folder-per-person tree,
and `libreyolo compare`, aliased as `libreyolo verify`, checks two images by
cosine similarity. See the [CLI reference](/docs/cli).

### Speed

CUDA graph capture of the training step went from 2 families to 24, across
detect, classify, semantic, point and restore. Measured on an RTX 5070 Ti under
AMP: 3.63x on FOMO, 2.74x on MobileNetV4, 1.99x on YOLO9-t, and 1.04x to 1.26x
for the rest, with the win tracking how much of a step is network rather than
loss. End to end on a 20-epoch YOLO9-t fine-tune of 406 images, dataloader and
validation included, 428 s became 368 s with identical mAP50-95 and per-epoch
losses.

Capture at prediction time reaches 39 families, spanning detect, segment, pose,
point, classify, semantic, depth, restore, matte and OCR. Every enabled family
is verified to replay bit-identically against two probe inputs. A family that
cannot be captured whole is split at a verified seam, and the remainder runs
eagerly with identical numbers.

`pip install libreyolo[hub-kernels]` opts every Deformable-DETR-lineage family
into a compiled Apache-2.0 CUDA kernel for multi-scale deformable attention,
pinned to an audited revision. It applies to eager CUDA fp32 only; exports keep
the portable path.

Fused scaled dot product attention now runs across the transformer families
using stock torch, with no optional dependency. Measured on an RTX 5070 Ti
under fp16 autocast, roughly 1.8x on Swin window attention and 3.7x on OWLv2
vision attention. Families held to a byte-exact parity bar keep manual
attention by default and opt in explicitly, and export graphs keep the
primitive-op equation either way.

COCO metrics moved to faster-coco-eval by default: 15.6x faster overall and 56x
on detection-dense datasets, decided on measured parity across all 100 RF100-VL
test splits, where 1381 of 1400 metric values were bit-identical to
pycocotools, the maximum deviation was 2.22e-16, and headline deltas were
exactly 0. `--no-faster-coco-eval` opts out, and pycocotools stays the
automatic fallback when the package is missing.

### Training and validation

Training from scratch (`pretrained=False`) now works for every g0, g1 and g2
family through a seeded random init. Previously only yolo9, rfdetr and dfine
had a scratch path, and the other families silently loaded the pretrained
checkpoint anyway.

Rectangular input arrived: `imgsz=480x640` for prediction and validation
everywhere it makes sense, and rectangular *training* for the seven CNN detect
families, yolo9, yolo9_e2e, yolo9_p2, yolox, yolo7, rtmdet and picodet.
Transformer families and non-detect tasks raise a clear error, and both
dimensions must divide the family stride. Autobatch and DDP spawn honor the
rectangular size when probing.

`amp_dtype` selects bfloat16 for [training](/docs/train) and
[validation](/docs/train/validation), on `TrainConfig`, `ValidationConfig` and
as `--amp-dtype` on `train`, `val` and `profile`. GradScaler is skipped for
bf16. Validation caps are configurable too: `max_det` defaults to 300, and
`eval_max_det` decouples the COCO evaluator cap from NMS.

`val_loss=True` reached every trainable family that can support it, across
detect, classify, semantic and restore rather than the flagships alone. It
moved from `YOLO9Config` and `RFDETRConfig` to `TrainConfig`, so a family that
has not implemented it now raises a clear error instead of ignoring the flag.
Denoising terms are never included, because validation forwards without ground
truth.

Comet, ClearML, Neptune and DVCLive joined the built-in training
[loggers](/docs/train/loggers), with the same canonical metrics and
failure-isolation contract as the existing TensorBoard, MLflow and Weights &
Biases integrations.

The kernel registry moved to `libreyolo/kernels/`, organized by purpose.
`LIBREYOLO_KERNELS` replaces `LIBREYOLO_QUANT_KERNELS`, which is still honored,
and `libreyolo.quant.kernels` remains a working alias.

### Predict sources

Webcams by index, RTSP and HTTP streams, screen capture, and `s3://` and
`gs://` URLs, with `--stream`, `--stream-buffer`, `--vid-stride` and `--show`.
Live sources implicitly enable streaming, which emits one JSON record per
frame. See [prediction sources](/docs/predict/sources).

### Fixes

`train(..., cuda_graph=True)` had been a silent no-op for D-FINE, DEIM, DEIMv2,
RT-DETRv4 and EC, whose trainers called the eager forward instead of the routed
one, so capture never engaged. The end-to-end suite now asserts that capture
actually engages per family.

YOLOX applies its BatchNorm `eps=1e-3` and `momentum=0.03` at construction
instead of as a fixup afterwards, so the values survive the class-count rebuild
that `train()` performs when a dataset's class count differs from the
checkpoint. A fine-tune previously trained and reported in-training validation
at torch's default `eps=1e-5` but was reloaded for inference at `1e-3`. On
RF100-VL `ball`, the same nano checkpoint scores 0.566 mAP50-95 evaluated at
its trained eps and 0.151 after a stock reload. Checkpoints trained before this
fix carry the old semantics and need the override described on
[upgrading](/docs/upgrade).

Non-strict checkpoint loads now warn with the counts and first names of missing
and unexpected state-dict keys. Shape mismatches always raised, but name
mismatches let a partially matching checkpoint load and then predict with
freshly initialized tensors, leaving no trace.

Auto-conversion no longer risks the file it writes: the new `.pt` is staged and
atomically renamed rather than written in place, and the original file mode is
preserved instead of collapsing to owner-only. Interrupted weight downloads
record a validator for `If-Range` resume and take a cross-process lock, so a
changed remote file or a second process can no longer produce corrupt weights.

CUDA graph capture no longer races with DataLoader pin-memory threads, which
had killed runs with a capture error from the pin-memory thread, and graphs are
captured and replayed on the right device. D-FINE training now applies
upstream's per-size multi-scale recipe instead of a hardcoded
`base_size_repeat=3`.

Several families silently produced wrong geometry and no longer do: PicoDet
clipped boxes against swapped axes, RTMDet segmentation resized masks to a
square derived from width only, Faster R-CNN ONNX exports placed boxes wrong on
non-square images, LW-DETR could drop real detections by running top-K over
unmapped COCO columns, and an exported MoGe-2 model run at an aspect ratio
other than its export canvas stretched the image into wrong normal directions
and now raises instead.

### Contributors

**juni3227**, first-time contributor: rectangular input resolution for the
convolution-based detectors, and a fix for training with YOLO-format datasets
at rectangular model sizes (#649, #658). **Xuban Ceccon**, maintainer:
everything else.

### Release stats

600 commits, 902 files changed, +125,701 / -3,315 lines, 88 merged pull
requests, 8 issues closed. 174 new test modules, 279 to 453. 38 new
documentation pages.

## 1.4.0, released 2026-07-24

The release notes summarize it as 15 new model families, 3 new tasks, a
quantization stack, two new trackers, and a multi-GPU training correctness
overhaul.

The new families were SegFormer, SwinIR, Real-ESRGAN, BiRefNet, ZipDepth,
Depth Anything 3, PP-OCRv5, SigLIP 2, YOLOv1, SAM 3, EdgeTAM, PicoSAM3,
OMDet-Turbo, OV-DEIM and SenseNova-Vision. The new tasks were `panoptic`,
`matte` and `ocr`, each with its result type and validator, and EoMT gained
instance and panoptic segmentation.

Quantization arrived as the `libreyolo quantize` command and `model.quantize()`,
with fp16, bf16, fp8, int8, w4a16, w4a8, nvfp4 and mxfp4 recipes, and
quantization-aware training through `train()` on a quantized checkpoint, for
yolo9 and rfdetr. Tracking gained BoT-SORT and Deep OC-SORT with an OSNet-AIN
re-identification embedder.

The multi-GPU work was about correctness. DDP now shards correctly for DEIM,
D-FINE and YOLO-NAS-pose, where every rank had been training the full dataset
at the full batch, and loss normalizers are globally all-reduced to match
single-GPU gradients. SyncBatchNorm defaults on for the CNN detectors, and a
non-divisible global batch or a custom loader that does not shard now raises at
setup instead of running wrong.

Several training fixes changed results outright: RTMDet fine-tune collapse from
missing head init took an nc=1 rebuild from 0.26 to 0.709 mAP50-95, YOLOv7
training color-space and fp16 overflow bugs took a fine-tune from 0.0 to 0.92
mAP50, and PicoDet's and DEIM's learning-rate defaults were lowered because the
old ones destroyed pretrained weights. Segmentation training no longer exhausts
RAM on COCO-scale datasets with multiple workers.

One compatibility note from that release: checkpoints written with the new task
strings or with finalized quantization state cannot be loaded by 1.3.1.

## Earlier releases

Releases before 1.4.0 are documented in the
[GitHub releases](https://github.com/LibreYOLO/libreyolo/releases) only, not in
`CHANGELOG.md`. Their documentation is still online, listed on
[versions](/docs/versions).

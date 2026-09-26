---
title: Changelog
seo_title: "LibreYOLO changelog: recent releases"
description: "What landed in LibreYOLO 1.5.0 and what 1.4.0 contained: 28 new model families, four new tasks, five export formats, torch-free ONNX inference and faster COCO metrics."
lead: "A summary of the repository changelog: what 1.5.0 shipped, and what 1.4.0 shipped before it."
keywords: [libreyolo changelog, libreyolo release notes, libreyolo 1.5.0, libreyolo 1.4.0, libreyolo new models]
last_verified: "1.6.0"
---

## 1.6.0, release preparation

These notes describe library dev at `4da12d005eb41a8e694a86124c97ea5faaee9b7c`. The v1.6.0 tag and release date are pending. See [upgrading](/docs/upgrade) for migrations.

### Highlights

<!-- H01 -->
- **Three new tasks. (#845, #846, #847, #848, #850, #857, #858)**
  3D boxes, albedo and robot action chunks join the existing result contracts.

<!-- H02 -->
- **More detection, pose, semantic and classification models. (#752, #759, #750, #745, #751, #836, #868)**
  PP-YOLOE, TinyFormer, DEKR, YOLO-NAS oriented boxes, PP-LiteSeg, U-Net and ConvNeXt V2 expand the model choices.

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
- **Training control. (#812, #875, #900, #782, #819)**
  Class filtering, custom fitness, opt-in training helpers and guardless local DDP extend existing training paths.

<!-- H08 -->
- **Data and deployment workflows. (#789, #780, #865, #893, #897)**
  Hub and FiftyOne integration, event histograms, TFLite INT8 I/O and expanded result plotting are available.

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
  the corresponding family checkpoint adds inference-only COCO-17 multi-person pose without a separate person detector, using the released `no_dc` graph at 640. `Results.boxes` are fitted to confident decoded joints. ONNX, TorchScript, TensorRT and OpenVINO export paths emit heatmaps and offsets for decoding. Weights remain linked to the source CDN.

<!-- M05 -->
- **YOLO-NAS gains oriented-box training and inference. (#745)**
  the corresponding family checkpoint uses a 1024 canvas and 18 DOTA2 classes, returning the existing `Results.obb` payload. Training uses rotated assignment/losses and `metrics/mAP50-95(OBB)` for checkpoint selection; `load_detect_weights_for_obb()` initializes from detection weights. ONNX, TorchScript, TensorRT and OpenVINO paths are available. Augmentation is flips and HSV. Pretrained weights remain non-commercial and non-redistributable, downloaded from the source CDN.

<!-- M06 -->
- **PP-LiteSeg semantic segmentation. (#751)**
  the corresponding family checkpoint adds trainable Cityscapes segmentation. Evaluation uses 512×1024 for `t50`/`b50` and 768×1536 for `t75`/`b75`; default training crops are 512×1024 and 768×768 respectively. The recipe includes auxiliary heads, Dice/cross-entropy/edge loss, polynomial learning-rate decay and EMA. ONNX, TorchScript, TensorRT and OpenVINO support the native rectangles.

<!-- M07 -->
- **U-Net semantic segmentation. (#836)**
  the corresponding family checkpoint identifies the trainable same-padded UNet-S5-D16 graph with FCN head: 19 Cityscapes classes, 1024×2048 evaluation and 512×1024 training crops, with cross-entropy plus auxiliary loss. Export is unsupported.

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
  the corresponding family checkpoint colorizes images; `LibreHVICIDNett-restore.pt` enhances low light with `gamma=1.0`, `saturation=1.0` and `intensity=1.0`; `LibreLaMab-restore.pt` requires `mask=` for inpainting; `LibreViTMattes-matte.pt` requires `trimap=` for matting. Python and predict CLI accept the single-image guides. All four support paired validation and reject training/export. LaMa runs an embedded ONNX graph.

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
- **Qwen3-VL detection fine-tuning. (#767, #834)**
  `LibreVLM("qwen3-vl-2b").train(data=...)` adds LoRA by default, optional full language-model fine-tuning with the vision tower frozen, checkpoint-directory reload, callbacks/loggers and validation-loss checkpoint selection. Defaults include `epochs=10`, `batch=1`, `accumulate=8`, `workers=0`, `hflip=0.5` and gradient checkpointing. `lr0=None` resolves to `1e-4` for LoRA or `2e-5` for full language-model tuning. Runs default to `runs/vlm/train` with `exist_ok=False`. Optimizer-state resume, COCO-JSON training and detection-mAP validation are unsupported.

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
- **F1-optimal detection confidence thresholds. (#778)**
  Detection validation adds `metrics/best_conf`, `metrics/best_conf_f1` and `metrics/best_conf_per_class`, computed at IoU `0.50`. Global selection uses micro F1; per-class thresholds are keyed by class name. Equal-score detections are grouped, and F1 ties choose the higher threshold. No positive F1 yields `NaN`. Segmentation's separate metrics implementation does not expose these keys.

<!-- A05 -->
- **Configurable validation sample plots. (#880)**
  `plot_samples=8` sets the retained sample-image budget on validation, training configuration and both CLI commands. `0` disables sample images and `-1` retains all. It does not limit evaluated images or the separate `visualize=True` output.

<!-- A06 -->
- **Two-channel event-histogram detection. (#865)**
  YOLO9 and RF-DETR detection accept prepared `(H, W, 2)` `.npy` arrays containing finite, nonnegative positive/negative counts. Dataset `input_profile` declares `format="event_histogram"`, `layout="HWC"`, `polarity="positive_negative"`, `encoding="counts"`, positive `scale` and positive integer `window_us`. Preprocessing clips to `scale`, divides by it and resizes without RGB normalization. RGB-weight initialization sets both input channels to `1.5 * mean(R,G,B)`; checkpoints preserve profile and initialization metadata. Native train/val/predict and FP32 ONNX without embedded NMS are supported. This path requires fixed-batch, single-device training and excludes live event decoding, video, TTA, tiling, LoRA, distillation and quantization.

<!-- A07 -->
- **Custom tracker instances. (#869)**
  `track(tracker=instance)` accepts the exported `libreyolo.tracking.Tracker` protocol: `reset()` and `update(results, image=None)`. Each run resets once and passes the original RGB PIL frame. Returned `track_id` must be a one-dimensional integer array/tensor aligned with boxes and on the matching backend/device. Custom instances use `track_conf=0.25` by default; configure them directly instead of passing `tracker_config` or tracker kwargs.

<!-- A08 -->
- **Tracking across image sequences. (#825)**
  `track()` accepts individual images, filename-sorted folders, lists, tuples and lazy image iterators as consecutive frames. New `fps=30.0` and `color_format="auto"` control timing and input interpretation. `vid_stride` reduces the output rate to `fps / vid_stride`; ByteTrack and BoT-SORT use that retained rate unless explicitly configured.

<!-- A09 -->
- **Hub checkpoint loading and publishing. (#789)**
  `LibreYOLO("owner/repo")` or `LibreYOLO("hf://owner/repo@revision/path/model.pt")` loads schema-tagged checkpoints through the Hub cache; existing local paths take precedence. `model.push_to_hub("owner/repo", private=False)` publishes `model.pt` and a model card. `HuggingFaceHubLogger` or `loggers="hf:owner/repo"` uploads `weights/best.pt`, falling back to `last.pt`, at training end; the logger defaults to `private=True`. Ambiguous repositories require an explicit filename; a sole `.pt` takes precedence over `.safetensors` candidates.

<!-- A10 -->
- **FiftyOne model application and dataset exchange. (#780)**
  `libreyolo.integrations.fiftyone` exports `apply_model`, `to_fiftyone_model`, `to_fiftyone` and `from_fiftyone`. Model application supports detect, segment, pose, OBB and classify. Dataset exchange covers detection/segmentation YOLO layouts and native COCO annotations, with curated views exportable to training YAML. `apply_model` defaults to `label_field="predictions"`, `conf=0.25`, `iou=0.45`, `max_det=300`, `mask_format="mask"` and `skip_failures=True`.

<!-- A12 -->
- **TFLite INT8 inputs and outputs for YOLOX and YOLO9 detection. (#893)**
  `export(format="tflite", int8=True, data=..., fraction=1.0, batch=1)` creates separate normalized-box and score outputs with independent quantization scales. The backend uses sidecar `output_layout` metadata to reconstruct predictions. Missing `data` falls back to `coco8.yaml` with a warning. Requires `onnx2tf[tensorflow]`; `half=True`, `dynamic=True`, other INT8 families/tasks and batch sizes other than 1 are rejected. Some internal operators may remain floating point.

<!-- A13 -->
- **Expanded `Results.plot()` rendering. (#897)**
  Detection, masks, pose, OBB, classification top-5, points, OCR, semantic/panoptic results and other image-overlay payloads use a shared renderer. Overlay output is contiguous `HxWx3` uint8 BGR; `pil=True` requests PIL. The existing edge/normal map paths keep PIL defaults. New controls include `img`, `conf=True`, `labels=True`, `boxes=True`, `masks=True`, `probs=True`, `line_width=None`, `pil=None`, `show=False`, `save=False` and `filename=None`. `orig_img` retains BGR pixels for in-memory/URL sources; local files and collected local finite-video/GIF frames can be reopened on demand.

<!-- T01 -->
- **Class-subset and single-class detection training. (#812, #875, #838)**
  `train(classes=[...])` filters supervision while preserving original class IDs and dataset `nc`/`names`; `single_cls=True` collapses retained labels to class 0, named `object`. Defaults are `classes=None`, `single_cls=False`. Both options support supported detector detection, including YOLO9 and RF-DETR; resume and validation inherit saved settings. CLI train/val expose class subsets. The public OBB dataset loader can collapse classes, but model OBB training rejects `single_cls`.

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
- **Guardless automatic local DDP. (#819)**
  `model.train(device=[0,1])` and `device="0,1"` use coordinator-managed ranks without replaying an ordinary unguarded training script's top-level code. Guarded scripts and explicit `torchrun` remain supported. Callback/logger objects requiring the documented standard-pickle fallback still need a main guard.

<!-- T11 -->
- **RF-DETR multi-class pose. (#888)**
  Dataset `kpt_names`, keyed by class index or name, retains the first `len(names)` keypoint rows per class; `[]` marks a box-only class. Head resizing and checkpoints retain `num_keypoints_per_class`, and predictions pad keypoints to `kpt_shape`, zeroing box-only classes. Multiclass datasets require `names`, and at least one class must have keypoints. Default keypoint-mAP fitness does not score box-only classes.

<!-- T12 -->
- **Rectangular semantic training datasets. (#751)**
  `SemanticDataset` accepts `(height, width)`, adds `rescale_crop` sampling with ignore padding and supports a family-provided `photometric` transform. The shared `PolyLRScheduler` supports the semantic recipes.

### Changed

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
  `predict(save=True)` and exported-backend saving use the shared result renderer, including tracking IDs. Matte saving still produces an RGBA cutout, while `plot()` provides a preview.

<!-- C08 -->
- **Classification augmentation can close for the final epochs. (#881)**
  `no_aug_epochs` disables `auto_augment`, `erasing`, `mixup` and `cutmix` in the final training tail while retaining crop and flips. Classification family configurations default that tail to `0`. Existing classifier helper imports remain re-exported from `data.classify_dataset` after the implementation moves to `data.augment.classify`.

<!-- C09 -->
- **Four detector recipes enable AMP by default. (#754)**
  D-FINE, DEIM, RT-DETRv4 and YOLO-NAS detection now use `amp=True`; `amp_dtype="float16"` is unchanged. Dome-DETR, PP-YOLOE and YOLO-NAS OBB retain FP32 defaults.

<!-- C10 -->
- **YOLO9 training recipe and checkpoint geometry. (#796)**
  New stock detection fine-tunes use a training-only PGI branch with `aux_weight=0.25`; `max_labels` rises from 100 to 300 and SGD momentum warms from `0.8` to `0.937` over three epochs. Old single-head checkpoints resume as single-head. Unmarked checkpoints keep `letterbox_pad="topleft"`; new official conversions stamp `"center"`, and `letterbox_pad=None` inherits the checkpoint setting. Inference/export use the main head.

<!-- C11 -->
- **RF-DETR and DINOv2 use incremented training directories. (#834)**
  Their Python `output_dir` default changes from `"runs/train"` to `None`, resolving to `runs/train/rfdetr_exp` and `runs/train/dinov2_exp`, with `exist_ok=False`. DINOv2's CLI default name is `dinov2_exp`; `resume=True` preserves the selected run directory.

<!-- C12 -->
- **Mosaic and MixUp prefer annotated partners. (#776)**
  YOLO9 and YOLOX mosaic use bounded partner sampling; YOLO9 MixUp does too. Up to 20 candidates are drawn, retaining the final draw if none have annotations. This changes the augmentation distribution for datasets containing empty images.

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
- **New stock YOLO9 fine-tunes use different training defaults. (#796)**
  PGI, the ground-truth cap and momentum warmup change new-run behavior.
  Migration: Use `aux_weight=0`, `max_labels=100` and `warmup_momentum=0.937` for the former choices. Set `letterbox_pad="topleft"` when starting from a new center-stamped conversion if the old geometry is required.

<!-- B06 -->
- **Default RF-DETR and DINOv2 artifact paths change. (#834)**
  Fresh Python runs use family-named, incremented directories.
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
- **RT-DETR filename detection no longer captures unrelated checkpoints. (#871)**
  Single-character `l` and `x` size codes require proper delimiters instead of matching names such as `last.pt` or `model_xlnet.pt`. Canonical RT-DETR filenames and the multi-character backbone-size prefix fallback remain accepted.

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
- **Classifier preprocessors supply calibration arrays. (#892)**
  AlexNet, VGG, ResNet, EfficientNetV2, ConvNeXt, MobileNetV4, DeiT, Swin and ViT now satisfy the calibration `(CHW array, ratio)` contract instead of dropping calibration images. DINOv2 classifier/embed calibration uses its classification pipeline instead of semantic preprocessing.

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

<!-- F12 -->
- **DINOv2 resume restores training state. (#834)**
  `resume=` now calls the trainer's resume path before continuing instead of only carrying the value in configuration.

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

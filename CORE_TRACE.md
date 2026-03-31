# Core Trace

本仓库基于以下核心源码进行最小化整理：

- `src/musubi_tuner/infer.py`
- `src/musubi_tuner/infer_network.py`

## 映射关系

- `infer.py::process_sample_prompts`
  - 对应：`open_infer/run_infer_json.py::encode_prompt_pair`
  - 保留 `prompt + control_img_list` 的图文编码逻辑

- `infer.py::do_inference`
  - 对应：`open_infer/run_infer_json.py::generate_two_frames`
  - 保留双 latent（`latents_f/latents_l`）拼接采样与双图解码

- `infer_network.py::load_prompts`
  - 对应：`open_infer/run_infer_json.py::process_json`
  - 采用 JSON list 作为输入

- `infer_network.py` 中 LoRA 动态挂载路径
  - `network_module -> create_arch_network -> apply_to -> load_weights`
  - 对应：`open_infer/run_infer_json.py::attach_dynamic_network_lora`

## 裁剪说明

为了可开源与可运行性，仓库仅保留推理需要的最小依赖闭包：

- 保留：Qwen image 推理主干、LoRA 动态挂载、必要工具模块
- 裁剪：训练脚本、数据训练流水线、大量与当前推理无关模块
- 轻量替换：
  - `dataset/image_video_dataset.py`：仅保留 bucket 计算与图像 resize 必需逻辑
  - `flux/flux_utils.py`：仅保留 `is_fp8`
  - `hv_generate_video.py`：仅保留 `synchronize_device`

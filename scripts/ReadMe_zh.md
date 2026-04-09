# Step-audio-editx 训练框架

本项目是基于TRL面向editx实现的训练框架，支持SFT、DPO和GRPO。

## 1. 环境准备

确保已安装以下依赖：
```bash
cd Step-Audio-EditX
uv sync --refresh
source .venv/bin/activate
```

## 2. 模型配置修改
在开始训练前，请检查您的模型文件夹。该目录应包含完整的模型权重及配置文件，文件结构如下：
```
Step-Audio-EditX/
├── CosyVoice-300M-25Hz
│   └── ...
├── config.json
├── configuration_step1.py
├── configuration.json
├── model-00001.safetensors
├── model.safetensors.index.json
├── modeling_step1.py
├── tokenizer_config.json
└── tokenizer.model
```

此外，必须手动修改配置文件：

- modeling_step1.py: 在340行中加入的`self.gradient_checkpointing = False`
```python
    ...
    def __init__(self, config):
        super().__init__(config)
        self.model = Step1Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.gradient_checkpointing = False
    ...
```

## 3. SFT训练
### 3.1 数据准备

数据格式为JSONL，每行包含以下字段。音频token提取可参考 `scripts/extract_audio_token_test.py`
```json
{
    "source_audio": "/path/to/audio.wav",
    "source_text": "原始音频文本内容",
    "source_vq02vq06": "source_audio音频token序列",
    "target_text": "目标音频文本内容",
    "target_vq02vq06": "目标音频token序列",
    "task_type": "edit",
    "edit_type": "emotion",
    "edit_info": "具体情绪类别",
}
```

### 3.2 训练启动
修改 `./scripts/run_edit_sft.sh`中的关键路径参数。
```bash
#!/bin/bash

# 模型与数据路径配置
MODEL_PATH="{EDITX_PATH}"                # Step-audio-editx 训练模型路径
DATA_FILES=(
    "{TRAINING_INDEX_FILE}"               # 训练数据索引文件路径 (JSONL)
    # 可以添加多个文件...
)

# 输出与日志配置
OUTPUT_DIR="{YOUR_PATH_TO_SAVE_CHECKPOINT}"                  # Checkpoint 保存路径
LOG_ROOT="{YOUR_PATH_TO_LOG_TRAINING_PROCESS}"                     # 训练日志存放路径
CONFIG_PATH="./config/train_config/accelerate_configs/deepspeed_zero2.yaml"
...
```
运行
```bash
bash ./scripts/run_edit_sft.sh
```

## 4. DPO训练
### 4.1 数据准备


数据格式为JSONL，每行包含以下字段。音频token提取可参考 `scripts/extract_audio_token_test.py`
```json
{
    "source_audio": "/path/to/audio.wav",
    "source_text": "原始音频文本内容",
    "source_vq02vq06": "source_audio音频token序列",
    "target_text": "目标音频文本内容",
    "target_vq02vq06": "chosen音频token序列",
    "rejected_vq02vq06": "rejected音频token序列",
    "task_type": "edit",
    "edit_type": "emotion",
    "edit_info": "具体情绪类别",
}
```


### 4.2 训练启动
修改 `./scripts/run_edit_dpo.sh`中的关键路径参数后，运行

```bash
bash ./scripts/run_edit_dpo.sh
```

## 5. GRPO训练

### 5.1 数据准备

数据格式为JSONL，每行包含以下字段。音频token提取可参考 `scripts/extract_audio_token_test.py`
```json
{
    "source_audio": "需要Edit的音频路径",
    "source_text": "source_audio对应的文本",
    "source_vq02vq06": "source_audio音频token序列",
    "target_text": "source_audio对应的文本",
    "task_type": "edit",
    "edit_type": "emotion",
    "edit_info": "具体情绪类别",
}
```


### 5.2 部署 Flow-matching 推理服务

本项目将Flow-matching部分与训练逻辑解耦。在启动训练脚本前，必须先部署 Flow-matching 推理服务。

**第一步：配置模型路径**

在启动服务前，请修改 `src/utils/flow_server.py` 文件第 15 行的 `FLOW_PATH` 变量，将其修改为您的 **CosyVoice-300M-25Hz** 本地模型路径。

```python
# src/utils/flow_server.py
FLOW_PATH = "/path/to/your/pretrained_models/CosyVoice-300M-25Hz"
```

**第二步：启动服务脚本**

配置完成后，执行以下命令启动推理服务（Flow-matching Server）：

```bash
cd ./src/utils/
bash run_server.sh
```

### 5.3 定义reward function

#### 5.3.1 自定义reward

您可以基于 `reward_func_genrm.py` 快速接入任意具有音频理解能力的多模态LLM 作为 reward model. 

**步骤 1：实现 API 调用接口**
打开 `src/utils/reward_func_genrm.py`，找到 `CustomGenerativeRM` 类。您只需要实现 `call_model` 方法，完成对您的目标模型 API 的调用即可。

**步骤 2：注册并启用新 Reward**
完成后需要在训练脚本`src/train_edit.py`中注册您的函数。
```python
# src/train_edit.py

# ... 引入您的函数
from utils.reward_func_genrm import genrm_reward_func

def main():
    # ...
    # 在此处注册
    reward_registry = {
        ....
        # 新增注册项：键名即为启动脚本中调用的名字
        "my_genrm": genrm_reward_func, 
    }
    # ...
```

在训练脚本中，将新注册的名字加入 `REWARD_FUNCS`：
```bash
# ./scripts/run_edit_grpo.sh

REWARD_FUNCS="my_genrm"
```

#### 5.3.2 使用gemini作为reward model
`src/utils/reward_func_gemini.py`中实现了基于gemini的reward function。如需使用，请按照以下步骤操作：

**步骤 1：配置环境变量**
在终端中设置您的 Google API Key：
```bash
export API_KEY="您的_GEMINI_API_KEY"
```

**步骤 2：在训练脚本中启用**
在训练脚本中，将`REWARD_FUNCS`参数设定为"gemini"：
```bash
# ./scripts/run_edit_grpo.sh

REWARD_FUNCS="gemini"
```

#### 5.3.3 Token-level 局部一致性 GRPO
如果希望使用 token-level 的局部一致性奖励，GRPO 数据集中还需要提供 `target_vq02vq06`，用于将生成结果和目标 codec 单元做对齐比较。

示例 JSONL 如下：
```json
{
    "source_text": "包含口水词或语气词的原始文本",
    "source_vq02vq06": "source_audio 音频 token 序列",
    "target_text": "删除口水词后的目标文本",
    "target_vq02vq06": "目标音频 token 序列",
    "task_type": "edit",
    "edit_type": "mask",
    "edit_info": "empty"
}
```

项目内置了 4 个 token-level reward：

- `token_level_edit`
- `token_level_consistency`
- `token_level_follow`
- `token_level_length`

同时兼容历史别名 `mask_edit`、`mask_consistency`、`mask_follow`、`mask_length`。如果需要单独调节各 reward 权重，可在启动脚本中加入；模板脚本默认都设为 `1.0`：

```bash
--mask_reward_weights "token_level_edit:1.0,token_level_consistency:1.0,token_level_follow:1.0,token_level_length:1.0"
```

如果你只启用这些 token-level reward，则不需要额外部署 Flow-matching reward server。


### 5.4 训练脚本启动

在确认 Flow-matching 服务已成功启动（端口正常监听且无报错）并完成奖励函数定义后，即可开始训练。

项目提供了四个训练脚本。`run_edit_grpo.sh` 使用标准 Hugging Face 推理模式，`run_edit_grpo_vllm.sh` 使用 vLLM 进行推理采样；另外还新增了 `run_edit_grpo_token_level.sh` 和 `run_edit_grpo_token_level_vllm.sh`，用于 token-level 局部一致性 GRPO 训练。

这两个 token-level 脚本刻意保持为保守模板，像 `num_generations` 这类参数默认设为 `1`，需要时由使用者自行调整。

在执行脚本前，请根据实际环境修改 `./scripts/` 目录下的脚本参数：
```bash
#!/bin/bash

# 模型与数据路径配置
MODEL_PATH="{EDITX_PATH}"                # Step-audio-editx 训练模型路径
DATA_FILES=(
    "{TRAINING_INDEX_FILE}"               # 训练数据索引文件路径 (JSONL)
    # 可以添加多个文件...
)

# 输出与日志配置
OUTPUT_DIR="{YOUR_PATH_TO_SAVE_CHECKPOINT}"                  # Checkpoint 保存路径
LOG_ROOT="{YOUR_PATH_TO_LOG_TRAINING_PROCESS}"                     # 训练日志存放路径
CONFIG_PATH="./config/train_config/accelerate_configs/deepspeed_zero2.yaml"

# 奖励函数与服务端配置
REWARD_FUNCS="my_genrm"                   # 启用的 Reward Function 名称
SERVER_IP="127.0.0.1"                     # Flow-matching 推理服务的 IP 地址
...

```


### 5.5 关键训练参数说明

| 参数名 | 默认值/示例 | 说明 |
| :---- | :--- | :--- |
| `--num_generations` | `8` | **组采样数 (Group Size)**。GRPO 算法核心参数，指对同一个 Prompt 并行生成的样本数量。|
| `--per_device_train_batch_size` | `2` | 单卡 Batch Size。指输入的 Prompt 数量。|
| `--gradient_accumulation_steps` | `2` | 梯度累积步数。用于在有限显存下模拟更大的 Batch Size 更新，以稳定训练梯度。 |
| `--max_audio_tokens` | `1024` | 音频 Token 最大长度。控制生成的音频时长上限，过长会显著增加显存消耗和训练时间。 |
| `--reward_server_ip` | `{SERVER_IP}` | **Flow Server 地址**。即步骤 4.1 中启动的 Flow-matching 服务 IP（本机通常为 `127.0.0.1`）。 |
| `--reward_server_num` | `2` | 服务端并发实例数。指定训练脚本同时连接的 Flow-matching Server 数量。**需与 `run_server.sh` 中启动的 `NUM_SERVERS` 保持一致**。
| `--reward_funcs` | `...` | 启用的奖励函数列表|
| `--use_vllm` | `false` | **启用 vLLM 加速**。开启后将使用 vLLM 引擎进行推理采样，能大幅提升采样速度并优化显存分配。 |
| `--vllm_mode` | `"colocate"` | **部署模式**。通常设为 `colocate`。 |
| `--vllm_gpu_memory_utilization` | `0.3` | **推理显存占用比**。指定 vLLM 占用的显存百分比（0.0-1.0）。 |

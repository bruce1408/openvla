"""A small, runnable teaching implementation of the OpenVLA idea.

This file is NOT the real OpenVLA-7B implementation. It is a compact model that
keeps the same high-level algorithm:

    RGB image + language instruction -> action tokens -> continuous robot action

Simplifications compared with real OpenVLA:
- Vision backbone: a tiny ResNet-16-like CNN instead of DINOv2 + SigLIP.
- Language backbone: Embedding + GRU instead of Llama-2.
- Multimodal fusion: concat + MLP instead of a full Prismatic VLM projector/LLM.
- Autoregressive decoder: single-layer GRU instead of 32-layer Transformer,
  but keeps the key mechanism: prefill + decode loop, action tokens at the
  tail of the vocabulary, and token_id -> bin mapping.

The goal is to make the model structure, training loss, and inference decoding
clear enough to read and run on CPU.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class ToyOpenVLAConfig:
    """整个玩具模型的超参数集合。

    用 frozen dataclass 把所有维度/尺寸集中管理，避免在各个模块里散落魔法数字。
    改一个字段就能同步影响视觉、文本、融合、动作头的构建。
    """

    image_size: int = 224     # 输入图像会被 resize 成 image_size x image_size 的正方形（与官方 OpenVLA 一致）。
    vocab_size: int = 128     # 文本词表上限（含 <pad>/<unk>），真实 OpenVLA 用 Llama 的大词表。
    max_text_len: int = 32    # 指令 token 的固定长度，超过截断、不足用 <pad> 补齐。
    text_dim: int = 128       # 文本 embedding 维度。
    vision_dim: int = 128     # 视觉编码器输出的每个 patch 的特征维度。
    vision_patches: int = 16  # 视觉 patch 数量（4×4 网格），官方 OpenVLA 用 256 个 patch。
    fusion_dim: int = 256     # 统一 token 维度（对应 Llama 的 hidden_size=4096，所有 token 对齐到此维度）。
    action_dim: int = 7       # 动作维度：7-DoF（xyz 位移 + 姿态 + 夹爪）。
    action_bins: int = 256    # 每个动作维度离散化成多少个 bin（分类类别数）。
    min_action: float = -1.0  # 归一化动作的下界，动作离散化/反离散化的区间左端。
    max_action: float = 1.0   # 归一化动作的上界，动作离散化/反离散化的区间右端。


class SimpleTokenizer:
    """Tiny whitespace tokenizer for the teaching model.

    Real OpenVLA uses a Llama tokenizer. Here we keep only the mechanics needed
    to turn an instruction string into integer token ids.
    """

    pad_token = "<pad>"  # 补齐用的特殊 token，id 固定为 0。
    unk_token = "<unk>"  # 未登录词（词表里没有的词）统一映射到它，id 固定为 1。

    def __init__(self, vocab: dict[str, int] | None = None, max_len: int = 32) -> None:
        self.max_len = max_len
        
        # 词表：词 -> 整数 id。默认只放两个特殊 token，其余通过 build_vocab 填充。
        self.vocab = vocab or {self.pad_token: 0, self.unk_token: 1}

    @staticmethod
    def _words(text: str) -> list[str]:
        # 统一转小写，再用正则抽取“字母/数字/下划线”组成的词，等价于简单的分词。
        # 标点、空格会被自然丢弃。
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())

    def build_vocab(self, texts: Iterable[str], max_vocab_size: int) -> None:
        # 统计所有文本里每个词出现的次数。
        counts: dict[str, int] = {}
        for text in texts:
            for word in self._words(text):
                counts[word] = counts.get(word, 0) + 1

        # 按“词频降序、同频按字母升序”排序，保证词表构建结果是确定性的（可复现）。
        for word, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            if len(self.vocab) >= max_vocab_size:
                break  # 达到词表容量上限就停止，低频词会被丢弃（后续当作 <unk>）。
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)  # 用当前长度作为新词的 id，保证连续不重复。

    def encode(self, text: str) -> torch.Tensor:
        # 逐词查表转成 id，查不到的词用 <unk> 的 id 兜底。
        ids = [self.vocab.get(word, self.vocab[self.unk_token]) for word in self._words(text)]
        ids = ids[: self.max_len]  # 超长截断到 max_len。
        
        # 不足 max_len 的部分用 <pad> 的 id 补齐，保证每条指令长度一致，方便组 batch。
        ids += [self.vocab[self.pad_token]] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


class ActionTokenizer:
    """Discretizes continuous actions into bins and decodes bins back to floats.

    与官方 OpenVLA (prismatic/vla/action_tokenizer.py) 完全一致的实现：
    - bins = linspace(-1, 1, 256)  → 256 个边界点 → 255 个 bin_centers
    - encode 用 digitize 逻辑 → 返回 [1, 256]
    - decode 用 clip(discretized - 1, 0, 254) → 索引 255 个 bin_centers
    """

    def __init__(self, bins: int = 256, min_action: float = -1.0, max_action: float = 1.0) -> None:
        self.n_bins = bins
        self.min_action = min_action
        self.max_action = max_action

        # 官方: np.linspace(-1, 1, 256) → 256 个边界点（不是 257！）
        self.bins = torch.linspace(min_action, max_action, bins)

        # 官方: 256 个边界 → 255 个 bin 中心（相邻边界的中点）
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0  # 255 个值

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """连续动作 [-1,1] → bin 索引 [1, n_bins]。

        与官方 np.digitize 一致：返回值范围是 [1, n_bins]（1-indexed），
        不是 [0, n_bins-1]。这是 np.digitize 的语义：返回第一个大于 action
        的边界索引。

        官方代码 (action_tokenizer.py:41):
            discretized_action = np.digitize(action, self.bins)
        """
        actions = actions.clamp(self.min_action, self.max_action)

        # 复刻 np.digitize: 找到第一个 >= action 的 bin 边界索引
        # np.digitize(action, bins) 返回 i，使得 bins[i-1] <= action < bins[i]
        # 结果范围: [1, len(bins)] = [1, 256]
        # 注意: right=True 才能匹配 np.digitize 在边界 (action==±1.0) 的行为
        discretized = torch.bucketize(actions, self.bins, right=True)
        return discretized.clamp(1, self.n_bins)  # 确保在 [1, 256] 范围内

    def decode(self, action_bins: torch.Tensor) -> torch.Tensor:
        """bin 索引 [1, 256] → 归一化连续动作。

        与官方 decode_token_ids_to_actions 一致 (action_tokenizer.py:65-68)：
            discretized = clip(discretized - 1, 0, 254)  # [1,256] → [0,254]
            return bin_centers[discretized]               # 255 个中心点
        """
        
        centers = self.bin_centers.to(action_bins.device)
        
        # [1, 256] → [0, 255] → clip 到 [0, 254]（因为只有 255 个 bin_centers）
        idx = (action_bins - 1).clamp(0, self.bin_centers.shape[0] - 1)
        return centers[idx]

    @staticmethod
    def unnormalize(normalized_actions: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
        """把归一化到 [-1, 1] 的动作反归一化回数据集的真实物理尺度。

        官方公式 (openvla.py:97-100):
            actions = 0.5 * (normalized + 1) * (q99 - q01) + q01
        """
        q01 = q01.to(normalized_actions.device)
        q99 = q99.to(normalized_actions.device)
        return 0.5 * (normalized_actions + 1.0) * (q99 - q01) + q01


class ResidualBlock(nn.Module):
    """标准的 ResNet 残差块：两层 3x3 卷积 + 跳跃连接（shortcut）。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        
        # 第一层卷积负责改变通道数/下采样（由 stride 控制），bias=False 因为后面接 BN。
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 第二层卷积保持尺寸不变，进一步提取特征。
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 当尺寸或通道数发生变化时，跳跃连接不能直接相加，需要用 1x1 卷积把输入投影到相同形状。
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()  # 形状一致时跳跃连接是恒等映射。

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)             # 准备用于相加的残差分支。
        x = F.relu(self.bn1(self.conv1(x)))     # conv1 -> BN -> ReLU。
        x = self.bn2(self.conv2(x))             # conv2 -> BN（此处先不激活）。
        return F.relu(x + residual)             # 主分支与残差相加后再激活，这是 ResNet 的核心。


class TinyResNet16Backbone(nn.Module):
    """Small ResNet-like image encoder that outputs patch tokens (not a single vector).

    复刻官方 DINOv2+SigLIP 的核心思想：保留空间位置信息，输出多个 patch token，
    而非压缩成单个全局向量。官方输出 256 个 patch（每个 4096 维），这里简化为
    16 个 patch（4×4 网格，每个 vision_dim 维）。

    结构: 8 residual blocks × 2 conv = 16 层 → AdaptiveAvgPool2d(4,4) → 16 个 patch。
    """

    def __init__(self, output_dim: int, n_patches: int = 16) -> None:
        super().__init__()
        self.n_patches = n_patches
        # 4×4=16 个 patch，取 sqrt 得到网格边长。
        self.grid_size = int(n_patches ** 0.5)

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
        )
        # 关键改动：池化到 grid_size×grid_size 而非 1×1，保留空间位置信息。
        # 官方 DINOv2+SigLIP 输出 256 个 patch token，这里输出 16 个。
        self.pool = nn.AdaptiveAvgPool2d((self.grid_size, self.grid_size))
        self.proj = nn.Linear(128, output_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)          # [B,3,224,224] -> [B,32,112,112]
        x = self.blocks(x)             # -> [B,128,28,28]
        x = self.pool(x)               # -> [B,128,4,4]  (保留 4×4 空间网格)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B,128,4,4] -> [B,128,16] -> [B,16,128]
        return self.proj(x)            # [B, n_patches, output_dim]


class TinyTextEncoder(nn.Module):
    """Token embedding + projector, preserving the full token sequence.

    复刻官方 Llama 的 input embedding 层：token ID → embedding → projector 对齐到
    fusion_dim。不压缩成单个向量，保留每个 token 的独立表示，与视觉 patch token
    拼接成完整序列后送入 decoder。
    """

    def __init__(self, vocab_size: int, embed_dim: int, fusion_dim: int, pad_id: int = 0) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        # projector: 把 embedding 维度对齐到 fusion_dim（对应 Llama 的 input embedding → hidden_size）。
        self.proj = nn.Linear(embed_dim, fusion_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)   # [B, T] -> [B, T, embed_dim]
        return self.proj(embedded)             # [B, T, fusion_dim]


class ToyOpenVLA(nn.Module):
    """Minimal VLA model with autoregressive action token generation.

    复刻真实 OpenVLA 的三个核心机制：
    1. 动作 token 占据词表尾部（Llama 词表末尾 256 个位置是 action bin）。
    2. 动作通过自回归逐 token 生成（prefill + decode），而非并行分类。
    3. token_id → bin 映射：bin = effective_vocab_size - token_id - 1。
    """

    def __init__(self, config: ToyOpenVLAConfig) -> None:
        super().__init__()
        self.config = config
        self.vision = TinyResNet16Backbone(config.vision_dim, config.vision_patches)
        self.text = TinyTextEncoder(config.vocab_size, config.text_dim, config.fusion_dim)

        # 视觉 projector：把每个 patch 投影到 fusion_dim（对应官方的 projector MLP）。
        self.vision_projector = nn.Linear(config.vision_dim, config.fusion_dim)

        # 词表布局（复刻真实 OpenVLA）：
        #   [0, text_vocab_size) = 文本 token（对应 Llama 的 32000 个文本 token）
        #   [text_vocab_size, effective_vocab_size) = 动作 token（256 个 bin）
        self.text_vocab_size = config.vocab_size
        self.effective_vocab_size = config.vocab_size + config.action_bins

        # token embedding：动作 token 的 embedding（对应 LLM 的 input embedding，文本已由 TinyTextEncoder 处理）。
        self.token_embedding = nn.Embedding(self.effective_vocab_size, config.fusion_dim)

        # 自回归解码器（对应 LLM 的 transformer 层，这里用单层 GRU 简化）。
        self.decoder = nn.GRU(config.fusion_dim, config.fusion_dim, batch_first=True)

        # lm_head：映射到完整词表（对应 LLM 的 lm_head，输出包含文本+动作 token 的 logits）。
        self.lm_head = nn.Linear(config.fusion_dim, self.effective_vocab_size)

        self.action_tokenizer = ActionTokenizer(config.action_bins, config.min_action, config.max_action)

    def _build_multimodal_sequence(self, images: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """构建多模态 token 序列（复刻官方 build_multimodal_inputs 的拼接逻辑）。

        官方顺序 (common.py:117-124):
            [BOS] + [256 视觉 patch] + [剩余文本 token]
        Toy 顺序:
            [第1个文本token] + [16 视觉 patch] + [剩余31个文本token]
        = [1 + 16 + 31] = 48 个 token，对应官方的 [1 + 256 + 5] = 262 个 token。

        官方用文本序列的第一个 token (BOS) 作为序列起始标记，这里复用同一思路。
        """
        # 视觉 patch tokens → projector 对齐到 fusion_dim
        image_features = self.vision(images)                    # [B, n_patches, vision_dim]
        projected_patches = self.vision_projector(image_features)  # [B, n_patches, fusion_dim]

        # 文本 token embeddings（已由 TinyTextEncoder 对齐到 fusion_dim）
        text_embeds = self.text(token_ids)                      # [B, max_text_len, fusion_dim]

        # 拼接: [第1个文本token] + [视觉patch] + [剩余文本token]
        return torch.cat(
            [
                text_embeds[:, :1, :],        # [B, 1, fusion_dim]  - 对应官方 BOS
                projected_patches,             # [B, 16, fusion_dim] - 对应官方 256 视觉 token
                text_embeds[:, 1:, :],         # [B, 31, fusion_dim] - 对应官方 5 个文本 token
            ],
            dim=1,
        )  # [B, 48, fusion_dim]

    def _bin_to_token_id(self, bins: torch.Tensor) -> torch.Tensor:
        """bin 索引 [1,256] → token ID（复刻官方 action_tokenizer.py:45）。

        官方公式: token_id = vocab_size - discretized
          bins=1   → token_id = effective_vocab_size - 1  (词表最后一个 = 最小动作)
          bins=256 → token_id = effective_vocab_size - 256 (动作区起始 = 最大动作)
        """
        return self.effective_vocab_size - bins

    def _token_id_to_bin(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token ID → bin 索引 [1,256]（复刻官方 action_tokenizer.py:65）。

        官方公式: discretized = vocab_size - token_id
        注意：这里返回的是 [1, 256]，decode() 内部会 -1 并 clip 到 [0, 254]。
        """
        return (self.effective_vocab_size - token_ids).clamp(1, self.config.action_bins)

    def forward(
        self, images: torch.Tensor, token_ids: torch.Tensor, target_actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """训练前向：teacher-forcing 的 next-token 预测。

        Prefill:  构建多模态序列 [BOS]+[视觉]+[文本] → GRU → 最终 hidden state。
        Decode:   用 hidden 作为初始状态，teacher-forcing 预测 7 个 action token。
        """
        # --- Prefill: 处理多模态序列 ---
        multimodal_seq = self._build_multimodal_sequence(images, token_ids)  # [B, 48, fusion_dim]
        prefill_output, hidden = self.decoder(multimodal_seq)  # hidden: [1, B, fusion_dim]

        # --- Decode: teacher-forcing 预测动作 token ---
        target_bins = self.action_tokenizer.encode(target_actions)    # [B, action_dim]
        target_token_ids = self._bin_to_token_id(target_bins)         # [B, action_dim]

        target_embeds = self.token_embedding(target_token_ids)  # [B, action_dim, fusion_dim]

        # decoder 输入: 右移一位 (第一个输入用 prefill 最后一步的输出)
        prefill_last_output = prefill_output[:, -1:, :]  # [B, 1, fusion_dim]
        decoder_input = torch.cat(
            [prefill_last_output, target_embeds[:, :-1, :]],
            dim=1,
        )  # [B, action_dim, fusion_dim]

        decoder_output, _ = self.decoder(decoder_input, hidden)  # [B, action_dim, fusion_dim]
        logits = self.lm_head(decoder_output)  # [B, action_dim, effective_vocab_size]

        return logits, target_token_ids

    def loss(self, images: torch.Tensor, token_ids: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Next-token 交叉熵损失（只在动作 token 上计算）。

        对应真实 OpenVLA 的训练损失：next-token cross-entropy，
        模型需要学会在 action token 区间内预测正确的 bin。
        """
        logits, target_token_ids = self.forward(images, token_ids, actions)
        return F.cross_entropy(
            logits.reshape(-1, self.effective_vocab_size),
            target_token_ids.reshape(-1),
        )

    @torch.inference_mode()
    def predict_action(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        q01: torch.Tensor | None = None,
        q99: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """自回归动作生成（复刻真实 OpenVLA 的 prefill + decode 流程）。

        Prefill:  [BOS]+[视觉patch]+[文本] 序列 → GRU → hidden state（汇总全部上下文）。
        Decode:   hidden → lm_head → 生成第 1 个 action token → embedding → GRU → 第 2 个 → ...
                  每步的 GRU hidden state 携带历史信息，实现维度间依赖。
        """
        # --- Prefill: 处理多模态序列，得到初始 hidden state ---
        multimodal_seq = self._build_multimodal_sequence(images, token_ids)  # [B, 48, fusion_dim]
        prefill_output, hidden = self.decoder(multimodal_seq)  # hidden: [1, B, fusion_dim]

        # decode 的首步输入 = prefill 最后一步的输出
        decoder_input = prefill_output[:, -1:, :]  # [B, 1, fusion_dim]

        generated_token_ids: list[torch.Tensor] = []

        for step in range(self.config.action_dim):
            decoder_output, hidden = self.decoder(decoder_input, hidden)  # [B, 1, fusion_dim]
            logits = self.lm_head(decoder_output[:, -1, :])  # [B, effective_vocab_size]

            next_token_id = logits.argmax(dim=-1)  # [B]  贪心解码
            generated_token_ids.append(next_token_id)

            decoder_input = self.token_embedding(next_token_id).unsqueeze(1)  # [B, 1, fusion_dim]

        # token ID → bin → 连续动作
        token_ids_tensor = torch.stack(generated_token_ids, dim=1)  # [B, action_dim]
        bins = self._token_id_to_bin(token_ids_tensor)
        normalized_actions = self.action_tokenizer.decode(bins)

        if q01 is None or q99 is None:
            return normalized_actions
        return self.action_tokenizer.unnormalize(normalized_actions, q01, q99)


class SyntheticRobotDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Small synthetic dataset for demonstrating training mechanics.

    Each sample contains:
    - random RGB image tensor [3, H, W]
    - tokenized instruction
    - synthetic normalized action [7]

    The target action is deterministic-ish from the instruction and image mean,
    so the training loop has a meaningful target without needing robot data.
    """

    instructions = [
        "pick up the gray ball",
        "move the robot arm forward",
        "grasp the blue object",
        "place the object on the table",
        "open the gripper",
        "close the gripper",
    ]

    def __init__(self, tokenizer: SimpleTokenizer, config: ToyOpenVLAConfig, length: int = 128) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.length = length  # 数据集样本总数（虚拟生成，不占磁盘）。

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # 用 index 作为随机种子，保证同一 index 每次取到完全相同的样本（可复现）。
        generator = torch.Generator().manual_seed(index)
        image = torch.rand(3, self.config.image_size, self.config.image_size, generator=generator)
        
        # 指令在固定列表里循环选取。
        instruction = self.instructions[index % len(self.instructions)]
        token_ids = self.tokenizer.encode(openvla_prompt(instruction))  # 套上 OpenVLA prompt 模板再编码。

        # 下面构造“伪标签”动作：让目标动作和指令、图像存在确定性关系，
        # 这样即使没有真实机器人数据，训练也有可学习的规律。
        image_mean = image.mean().item()
        phase = (index % len(self.instructions)) / len(self.instructions)  # 把指令编号映射到 [0,1) 的相位。
        base = torch.tensor(
            [
                math.sin(phase * math.pi),                                   # 维度0：随相位变化的正弦分量。
                math.cos(phase * math.pi),                                   # 维度1：随相位变化的余弦分量。
                image_mean * 2.0 - 1.0,                                      # 维度2：由图像亮度均值映射到 [-1,1]。
                0.25 if "grasp" in instruction or "pick" in instruction else -0.25,  # 维度3：是否抓取类动作。
                0.5 if "forward" in instruction else -0.5,                   # 维度4：是否前进。
                0.75 if "place" in instruction else -0.75,                   # 维度5：是否放置。
                1.0 if "open" in instruction else -1.0 if "close" in instruction else 0.5,  # 维度6：夹爪开/合/其它。
            ],
            dtype=torch.float32,
        )
        
        # 返回 (图像, 指令 token, 归一化动作)，动作再裁剪一次确保落在 [-1,1]
        return image, token_ids, base.clamp(-1.0, 1.0)


def openvla_prompt(instruction: str) -> str:
    
    # 复刻 OpenVLA 的 prompt 模板，把裸指令包装成模型期望的问答格式。
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def load_image_tensor(path: str, image_size: int) -> torch.Tensor:
    
    # 读图 -> 转 RGB -> resize 成正方形。
    image = Image.open(path).convert("RGB").resize((image_size, image_size))
    
    # 把 PIL 原始字节读进 ByteTensor（形状暂时是一维）。
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    
    # reshape 成 [H, W, 3]，再调整成 PyTorch 期望的 [3, H, W]，并归一化到 [0,1]。
    data = data.view(image_size, image_size, 3).permute(2, 0, 1).float() / 255.0
    return data


def train_demo(model: ToyOpenVLA, dataset: SyntheticRobotDataset, steps: int, batch_size: int) -> None:
    
    # DataLoader 负责成批取样并打乱顺序。
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # AdamW 优化器。
    model.train()  # 切到训练模式（启用 BatchNorm 统计更新、dropout 等）。

    # zip(range(steps), loader) 保证最多只跑 steps 步，即使数据集更大也提前停止。
    for step, (images, token_ids, actions) in zip(range(steps), loader):
        optimizer.zero_grad(set_to_none=True)   # 清空上一步的梯度。
        loss = model.loss(images, token_ids, actions)  # 前向 + 计算交叉熵损失。
        loss.backward()                          # 反向传播求梯度。
        optimizer.step()                         # 按梯度更新参数。
        print(f"train step {step + 1}/{steps}: loss={loss.item():.4f}")


def run_prediction_demo(model: ToyOpenVLA, tokenizer: SimpleTokenizer, image: torch.Tensor, instruction: str) -> None:
    model.eval()  # 切到推理模式（关闭 dropout、用 BN 的运行统计）。
    prompt = openvla_prompt(instruction)
    
    # 编码后用 unsqueeze(0) 加上 batch 维，凑成 [1, max_text_len]。
    token_ids = tokenizer.encode(prompt).unsqueeze(0)
    
    # 图像同样加 batch 维成 [1,3,H,W]，预测一个 7 维动作。
    action = model.predict_action(image.unsqueeze(0), token_ids)
    print("prompt:", prompt)
    print("predicted normalized 7D action:", action[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy OpenVLA implementation for learning the model structure.")
    parser.add_argument("--image", default=None, help="Optional RGB image path. If omitted, uses a random image.")
    parser.add_argument("--instruction", default="pick up the gray ball")
    parser.add_argument("--train-steps", type=int, default=3)   # 演示用，默认只训练 3 步。
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = ToyOpenVLAConfig()
    tokenizer = SimpleTokenizer(max_len=config.max_text_len)
    
    # 用数据集里所有指令（套过 prompt 模板）来构建词表。
    tokenizer.build_vocab((openvla_prompt(text) for text in SyntheticRobotDataset.instructions), config.vocab_size)

    model = ToyOpenVLA(config)                          # 构建模型。
    dataset = SyntheticRobotDataset(tokenizer, config)  # 构建合成数据集。
    train_demo(model, dataset, steps=args.train_steps, batch_size=args.batch_size)  # 简短训练。

    # 有传图片就加载真实图片，否则用随机图像做推理演示。
    if args.image:
        image = load_image_tensor(args.image, config.image_size)
    else:
        image = torch.rand(3, config.image_size, config.image_size)
    run_prediction_demo(model, tokenizer, image, args.instruction)  # 跑一次预测并打印动作。


if __name__ == "__main__":
    main()

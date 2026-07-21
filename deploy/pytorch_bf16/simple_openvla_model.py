"""A small, runnable teaching implementation of the OpenVLA idea.

This file is NOT the real OpenVLA-7B implementation. It is a compact model that
keeps the same high-level algorithm:

    RGB image + language instruction -> action tokens -> continuous robot action

Simplifications compared with real OpenVLA:
- Vision backbone: a tiny ResNet-16-like CNN instead of DINOv2 + SigLIP.
- Language backbone: Embedding + GRU instead of Llama-2.
- Multimodal fusion: concat + MLP instead of a full Prismatic VLM projector/LLM.
- Action head: predicts one discrete bin per action dimension.

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

    image_size: int = 128     # 输入图像会被 resize 成 image_size x image_size 的正方形。
    vocab_size: int = 128     # 文本词表上限（含 <pad>/<unk>），真实 OpenVLA 用 Llama 的大词表。
    max_text_len: int = 32    # 指令 token 的固定长度，超过截断、不足用 <pad> 补齐。
    text_dim: int = 128       # 文本编码器（Embedding + GRU）的隐藏维度。
    vision_dim: int = 128     # 视觉编码器输出的图像特征向量维度。
    fusion_dim: int = 256     # 图文特征融合后 MLP 的隐藏维度。
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

    Real OpenVLA maps action bins onto the tail of the LLM vocabulary. This toy
    model predicts the bin index directly with a classification head, which is
    easier to inspect but equivalent for understanding training/inference.
    """

    def __init__(self, bins: int = 256, min_action: float = -1.0, max_action: float = 1.0) -> None:
        self.bins = bins
        self.min_action = min_action
        self.max_action = max_action
        # 在 [min_action, max_action] 上均匀切成 bins 段，需要 bins+1 个边界点。
        edges = torch.linspace(min_action, max_action, bins + 1)
        # 每个 bin 的中心值（相邻边界的中点），解码时用它作为该 bin 的代表动作值。
        self.bin_centers = (edges[:-1] + edges[1:]) / 2.0

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """把 [-1, 1] 区间的连续动作映射成整数 bin 索引。

        Args:
            actions: 形状 [batch, action_dim] 的浮点张量。

        Returns:
            形状 [batch, action_dim] 的 long 张量，取值范围 [0, bins - 1]。
        """

        actions = actions.clamp(self.min_action, self.max_action)  # 先裁剪到合法区间，防止越界。
        # 线性归一化到 [0, 1]。
        scaled = (actions - self.min_action) / (self.max_action - self.min_action)
        # 乘以 bins 并取整得到 bin 索引；再 clamp 一次，避免 action==max 时算出 bins（越界）。
        return torch.clamp((scaled * self.bins).long(), min=0, max=self.bins - 1)

    def decode(self, action_bins: torch.Tensor) -> torch.Tensor:
        """把整数 bin 索引还原成归一化的连续动作（取该 bin 的中心值）。"""

        centers = self.bin_centers.to(action_bins.device)  # 对齐设备，避免 CPU/GPU 张量混用报错。
        return centers[action_bins]  # 用高级索引一次性查出每个 bin 对应的中心值。

    @staticmethod
    def unnormalize(normalized_actions: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
        """把归一化到 [-1, 1] 的动作反归一化回数据集的真实物理尺度。

        这一步对应真实 OpenVLA 用训练数据集的动作分位数统计（q01/q99）来还原动作。
        本 demo 里 q01=-1、q99=1，所以反归一化实际上是恒等变换。真实机器人上
        这两个值必须来自对应机器人数据集的动作统计。
        """

        q01 = q01.to(normalized_actions.device)
        q99 = q99.to(normalized_actions.device)
        # 先把 [-1,1] 线性映射到 [0,1]，再拉伸到 [q01, q99] 区间。
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
    """Small ResNet-like image encoder used to replace OpenVLA's real vision stack.

    There is no widely used torchvision `resnet16` model, so this file implements
    a ResNet-16-like backbone for teaching: 8 residual blocks x 2 conv layers =
    16 residual conv layers. It outputs one fixed-size feature vector per image.
    """

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        # stem：入口卷积，把 3 通道 RGB 变成 32 通道，同时 stride=2 先做一次下采样。
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # 8 个残差块，每块含 2 层卷积 = 16 层，故称 “ResNet-16-like”。
        # 通道数按 32 -> 64 -> 128 逐步加深，stride=2 的块负责空间下采样。
        self.blocks = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),    # 下采样并升到 64 通道。
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),   # 下采样并升到 128 通道。
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
        )
        # 自适应平均池化到 1x1，把任意空间尺寸压成每通道一个值，得到全局特征。
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # 线性投影到指定的 output_dim（= vision_dim），便于后续与文本特征拼接。
        self.proj = nn.Linear(128, output_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)          # [B,3,H,W] -> [B,32,H/2,W/2]
        x = self.blocks(x)             # 经残差块提特征并下采样 -> [B,128,h,w]
        x = self.pool(x).flatten(1)    # 全局池化 -> [B,128,1,1] -> 展平 [B,128]
        return self.proj(x)            # 投影 -> [B, output_dim]


class TinyTextEncoder(nn.Module):
    """Small language encoder replacing Llama-2 in the teaching model."""

    def __init__(self, vocab_size: int, text_dim: int, pad_id: int = 0) -> None:
        super().__init__()
        # 词嵌入：把 token id 映射成 text_dim 维向量。padding_idx 让 <pad> 的嵌入恒为 0 且不参与训练。
        self.embedding = nn.Embedding(vocab_size, text_dim, padding_idx=pad_id)
        # 单层 GRU 顺序读入词向量序列，用最后的隐藏状态汇总整句语义。batch_first 表示输入形状为 [B, T, D]。
        self.gru = nn.GRU(text_dim, text_dim, batch_first=True)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)   # [B, T] -> [B, T, text_dim]
        # GRU 返回 (每步输出, 最终隐藏状态)；这里只要最终隐藏状态作为整句表示。
        _, hidden = self.gru(embedded)
        return hidden[-1]                      # hidden 形状 [num_layers, B, D]，取最后一层 -> [B, D]


class ToyOpenVLA(nn.Module):
    """Minimal VLA model: image + instruction -> action-bin logits."""

    def __init__(self, config: ToyOpenVLAConfig) -> None:
        super().__init__()
        self.config = config
        self.vision = TinyResNet16Backbone(config.vision_dim)          # 视觉编码器：图像 -> 特征向量。
        self.text = TinyTextEncoder(config.vocab_size, config.text_dim)  # 文本编码器：指令 -> 特征向量。
        # 融合模块：把拼接后的图文特征经两层 MLP 融合成联合表示（对应真实模型里 VLM 的作用）。
        self.fusion = nn.Sequential(
            nn.Linear(config.vision_dim + config.text_dim, config.fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.fusion_dim, config.fusion_dim),
            nn.ReLU(inplace=True),
        )
        # 动作头：一次性输出 action_dim * action_bins 个 logits，等价于对每个动作维做一次 bins 类分类。
        self.action_head = nn.Linear(config.fusion_dim, config.action_dim * config.action_bins)
        # 动作离散化/反离散化工具，训练时把连续动作转 bin，推理时把 bin 转回连续动作。
        self.action_tokenizer = ActionTokenizer(config.action_bins, config.min_action, config.max_action)

    def forward(self, images: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """返回动作 bin 的 logits，形状 [batch, action_dim, action_bins]。"""

        image_features = self.vision(images)       # [B, vision_dim]
        text_features = self.text(token_ids)       # [B, text_dim]
        # 沿特征维拼接图文特征，再经融合 MLP 得到联合表示。
        fused = self.fusion(torch.cat([image_features, text_features], dim=-1))
        logits = self.action_head(fused)           # [B, action_dim * action_bins]
        # reshape 成 [B, action_dim, action_bins]，最后一维就是每个动作维在各 bin 上的打分。
        return logits.view(-1, self.config.action_dim, self.config.action_bins)

    def loss(self, images: torch.Tensor, token_ids: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """动作 bin 的交叉熵损失：把每个动作维当成一次独立的分类任务。"""

        logits = self.forward(images, token_ids)              # [B, action_dim, action_bins]
        target_bins = self.action_tokenizer.encode(actions)   # 连续动作 -> 目标 bin 索引 [B, action_dim]
        # flatten(0,1) 把前两维合并成 [B*action_dim, ...]，让所有 (样本,动作维) 一起算交叉熵。
        return F.cross_entropy(logits.flatten(0, 1), target_bins.flatten(0, 1))

    @torch.inference_mode()
    def predict_action(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        q01: torch.Tensor | None = None,
        q99: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict a continuous 7-DoF action.

        Args:
            images: [batch, 3, H, W] float tensor in [0, 1].
            token_ids: [batch, max_text_len] instruction token ids.
            q01/q99: Optional per-action-dim statistics for unnormalization.

        Returns:
            [batch, action_dim] continuous action tensor.
        """

        logits = self.forward(images, token_ids)                    # [B, action_dim, action_bins]
        pred_bins = logits.argmax(dim=-1)                            # 每个动作维取分数最高的 bin（贪心解码）。
        normalized_actions = self.action_tokenizer.decode(pred_bins)  # bin 索引 -> 归一化连续动作。
        if q01 is None or q99 is None:
            return normalized_actions                               # 未提供统计量时直接返回归一化动作。
        # 提供了数据集统计量时，反归一化到真实物理尺度。
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
        # 返回 (图像, 指令 token, 归一化动作)，动作再裁剪一次确保落在 [-1,1]。
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

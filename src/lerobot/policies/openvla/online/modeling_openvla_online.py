import io
import requests
import torch
from torch import Tensor
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.openvla.configuration_openvla import OpenVLAConfig
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig  # <-- 加上 BitsAndBytesConfig


class OpenVLAPolicy(PreTrainedPolicy):
    config_class = OpenVLAConfig
    name = "openvla"

    def __init__(self, config: OpenVLAConfig, dataset_stats=None):
        super().__init__(config)
        self.config = config
        print("\n=======================================================")
        print(" 🚀[云脑-端控] 模式就绪！本机显存与内存消耗: 0 GB")
        print("=======================================================\n")

        # 【修改这里】：填入智星云分配的外网地址 + /predict
        self.cloud_api = "http://js3.blockelite.cn:14864/predict"

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *args, **kwargs):
        """【超级防爆内存拦截】直接拦截框架的读取逻辑，绝不去硬盘读那 15GB 的权重文件！瞬间秒进！"""
        config = cls.config_class.from_pretrained(pretrained_name_or_path)
        return cls(config)

    def reset(self):
        pass

    def get_optim_params(self):
        return {}

    def forward(self, batch):
        pass

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """核心契约函数：补齐框架要求的抽象方法，同时作为跨网发送器"""
        buffer = io.BytesIO()

        # 1. 剥离图片：打包摄像头的当前帧发过去
        clean_batch = {k: v.cpu() for k, v in batch.items() if isinstance(v, Tensor)}
        if "task" in batch:
            clean_batch["task"] = batch["task"]

        torch.save(clean_batch, buffer)

        try:
            # 2. 发起跨网请求，呼叫云端大脑 (超时时间设为 5 秒)
            res = requests.post(self.cloud_api, data=buffer.getvalue(), timeout=30)

            if res.status_code != 200:
                print(f"[警告] 云端大脑走神了，报错码: {res.status_code}")
                # 形状必须满足[Batch, Chunk_size, Action_dim]
                return torch.zeros((1, self.config.chunk_size, self.config.action_dim))

            # 3. 接收云端传回来的单步指令
            ret_buffer = io.BytesIO(res.content)
            action = torch.load(ret_buffer, map_location="cpu", weights_only=False)

            # 因为云端发回来的是 [Batch, Action_dim]，为了满足格式要求，强行加一个时间维度
            return action.unsqueeze(1)

        except Exception as e:
            print(f"[断网警报] 与云端服务器失去联系: {e}")
            return torch.zeros((1, self.config.chunk_size, self.config.action_dim))

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """测试脚本实际调用的动作抽离口"""
        actions = self.predict_action_chunk(batch, **kwargs)
        return actions[:, 0, :]
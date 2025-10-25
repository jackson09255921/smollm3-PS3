# In modeling_smolvlm_ps3.py

import torch
from torch import nn
import torch.nn.functional as F # 需要 F 用於插值
from transformers import Idefics3Model, Idefics3ForConditionalGeneration
# ... (其他 import) ...
from ps3 import PS3VisionModel

logger = logging.get_logger(__name__)

# --- 新增：定義卷積瓶頸塊 ---
class ConvBottleneck(nn.Module):
    """受 Convpass 和 C-Abstractor 啟發的卷積瓶頸塊"""
    def __init__(self, in_channels, mid_channels, out_channels, dw_kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
        # 使用 Depthwise Separable Convolution
        self.dw_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=dw_kernel_size, stride=1, padding=dw_kernel_size//2, groups=mid_channels, bias=False)
        self.norm = nn.LayerNorm([mid_channels, 1, 1], elementwise_affine=True) # 使用 LayerNorm
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv(x)
        # LayerNorm 需要 (B, C, H, W) -> (B, C, 1, 1) -> (B, C, H, W)
        h, w = x.shape[2], x.shape[3]
        x_norm = F.layer_norm(x.mean(dim=[2,3], keepdim=True), self.norm.normalized_shape, self.norm.weight, self.norm.bias, self.norm.eps)
        x = x + x_norm # 應用 LayerNorm 作為調整
        # 或者使用標準 LayerNorm:
        # x_reshaped = x.permute(0, 2, 3, 1) # B, H, W, C
        # x_norm = self.norm(x_reshaped)
        # x = x_norm.permute(0, 3, 1, 2) # B, C, H, W

        x = self.act(x)
        x = self.conv2(x)
        return x

class MyPS3VLMModel(Idefics3Model):
    def __init__(self, config):
        super().__init__(config) # 載入基礎 Idefics3 (包括 text_model)

        # --- 載入 PS3 Vision Encoder (保持不變) ---
        self.ps3_model_id = "nvidia/PS3-4K-SigLIP2"
        self.ps3_num_look_close = getattr(config, "ps3_num_look_close", 2)
        logger.info(f"正在從 {self.ps3_model_id} 載入 PS3VisionModel...")
        self.vision_model = PS3VisionModel.from_pretrained(
            self.ps3_model_id,
            cache_dir=getattr(config, "cache_dir", None),
            # 重要：讓 PS3 輸出多個層級的特徵
            output_hidden_states=True # 確保 PS3 配置支援此選項
        )
        # 獲取 PS3 相關維度 (假設最後一層維度為 D_ps3)
        D_ps3 = self.vision_model.config.hidden_size # 1152

        # --- 新增：定義從 PS3 提取哪些層級的特徵 ---
        # 根據 Deep Search 建議，例如 8, 16, 24 層
        # PS3 (ViT-L) 通常有 24 層 Transformer Block
        # 層索引通常從 0 開始，所以是 7, 15, 23 (以及最後一層 -1)
        # 你需要根據 PS3 實際的 `output_hidden_states` 結構調整索引
        self.ps3_feature_layers_indices = getattr(config, "ps3_feature_layers_indices", [7, 15, -1]) # 可配置
        num_feature_layers = len(self.ps3_feature_layers_indices)
        logger.info(f"將從 PS3 的第 {self.ps3_feature_layers_indices} 層提取特徵。")


        # --- 修改：實現空間感知密集連接器 ---
        D_llm = config.text_config.hidden_size # LLM 維度 (e.g., 500M 模型可能較小)

        # 步驟 3 的卷積瓶頸塊
        # 計算拼接後的通道數
        in_channels_connector = num_feature_layers * D_ps3
        # 設定中間和輸出通道 (可調整)
        mid_channels_connector = getattr(config, "connector_mid_channels", 1024) # 假設
        out_channels_connector = getattr(config, "connector_out_channels", D_llm) # 最終投影前的維度

        self.connector_conv_blocks = nn.Sequential(
             ConvBottleneck(in_channels_connector, mid_channels_connector, mid_channels_connector),
             # 可以堆疊更多塊
             # ConvBottleneck(mid_channels_connector, mid_channels_connector, mid_channels_connector),
             ConvBottleneck(mid_channels_connector, mid_channels_connector, out_channels_connector)
        )

        # 步驟 4 的最終線性投影
        # (現在包含在卷積塊的最後一個 1x1 卷積中了，如果 out_channels_connector == D_llm)
        # 如果 out_channels_connector != D_llm，則需要額外的線性層：
        # self.connector_final_proj = nn.Linear(out_channels_connector, D_llm) if out_channels_connector != D_llm else nn.Identity()

        # 將整個連接器（卷積+投影）賦值給 self.connector 以符合命名規範
        self.connector = self.connector_conv_blocks # (如果不需要額外的 final_proj)
        # self.connector = nn.Sequential(self.connector_conv_blocks, self.connector_final_proj) # (如果需要)

        logger.info(f"MyPS3VLMModel 初始化完畢：")
        logger.info(f"  Vision Encoder: PS3 (最後層維度: {D_ps3})")
        logger.info(f"  Connector: 空間感知密集連接器 (提取 {num_feature_layers} 層, 卷積處理)")
        logger.info(f"  LLM: {self.text_model.__class__.__name__} (輸入維度: {D_llm})")

    # --- inputs_merger (保持不變) ---
    # (從 modeling_smolvlm.py 完整複製)
    def inputs_merger(self, input_ids, inputs_embeds, image_hidden_states):
        # ... (完整程式碼) ...
        pass # Placeholder for brevity

    # --- 修改：Forward 方法以使用新連接器 ---
    def forward(self, input_ids, pixel_values=None, image_hidden_states=None, **kwargs):

        # ... (複製 forward 開頭的所有參數處理、use_cache、past_key_values 等邏輯) ...
        # ... (複製 input_ids/inputs_embeds 檢查邏輯) ...
        # ... (複製 get_input_embeddings 邏輯) ...

        # START VISUAL INPUTS INTEGRATION
        if pixel_values is not None and image_hidden_states is not None:
             raise ValueError("...")
        elif pixel_values is not None:
            # --- (複製 pixel_values 預處理： B,N,C,H,W -> B*N,C,H,W -> 過濾 padding -> real_pixels) ---
            # ... (pixel_values 處理程式碼) ...
            batch_size, num_images, num_channels, height, width = pixel_values.shape # 獲取原始 BS
            # ... (view, filter real_images_inds, ...)
            pixel_values_flat = pixel_values[real_images_inds].contiguous() # (N_real, C, H, W)

            # --- 修改：呼叫 PS3 並獲取多層特徵 ---
            vision_outputs = self.vision_model(
                pixel_values=pixel_values_flat,
                num_look_close=self.ps3_num_look_close,
                output_hidden_states=True # 確保請求 hidden_states
            )
            # vision_outputs.hidden_states 是一個包含所有層輸出的元組
            # 我們需要提取指定的層級
            # !! 假設 PS3 輸出包括輸入嵌入層，所以索引需要 +1 !! (需要確認 PS3 文檔)
            try:
                 selected_features = [vision_outputs.hidden_states[i+1] for i in self.ps3_feature_layers_indices]
            except (IndexError, TypeError):
                 logger.error("無法從 PS3 hidden_states 提取指定層級，請檢查索引或 PS3 輸出格式。")
                 # 可以選擇拋出錯誤或使用最後一層作為 fallback
                 selected_features = [vision_outputs.last_hidden_state]

            # --- 新增：實現連接器步驟 2, 3, 4 ---
            # 假設 PS3 輸出令牌是 (N_real, NumPatches, D_ps3)
            # 我們需要知道 H_patch, W_patch (NumPatches = H_patch * W_patch)
            # 這通常可以從 vision_model.config 或 patch_embedding 推斷
            # 假設我們可以得到 H_patch, W_patch
            num_patches = selected_features[0].shape[1]
            D_ps3 = selected_features[0].shape[2]
            # 嘗試自動推斷 H_patch, W_patch (假設是方形)
            H_patch = W_patch = int(num_patches**0.5)
            if H_patch * W_patch != num_patches:
                 raise ValueError("無法自動推斷方形 patch 網格，請手動指定 H_patch, W_patch。")

            reshaped_features = []
            for features in selected_features: # 對每個提取的層級
                 # 步驟 2: 空間重構 (N_real, NumPatches, D) -> (N_real, D, H_patch, W_patch)
                 features_2d = features.permute(0, 2, 1).reshape(-1, D_ps3, H_patch, W_patch)
                 # (可選) 如果層級間解析度不同，在此處進行插值
                 # features_2d = F.interpolate(features_2d, target_size=(H_target, W_target), mode='bilinear')
                 reshaped_features.append(features_2d)

            # 步驟 2: 拼接 (N_real, D*k, H_patch, W_patch)
            concatenated_features = torch.cat(reshaped_features, dim=1)

            # 步驟 3: 卷積瓶頸處理
            processed_features_2d = self.connector_conv_blocks(concatenated_features) # (N_real, D_out, H_out, W_out)

            # 步驟 4: 展平並投影 (如果需要)
            # (N_real, D_out, H_out, W_out) -> (N_real, H_out*W_out, D_out)
            processed_features_flat = processed_features_2d.flatten(2).permute(0, 2, 1)
            # image_hidden_states = self.connector_final_proj(processed_features_flat) # (N_real, S_new, D_llm)
            # 如果卷積塊直接輸出 D_llm，則不需要 final_proj
            image_hidden_states = processed_features_flat # (N_real, S_new, D_llm)

            # **** 重要：S (序列長度) 可能已經改變 ****
            # inputs_merger 假設 S 不變，我們需要調整 S
            # 這裡我們假設 S_new == S (卷積塊保持空間維度)
            # 如果 S 改變了， inputs_merger 需要修改或被完全替換

        elif image_hidden_states is not None:
             # (保持不變)
             image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=input_ids.device)

        # --- inputs_merger 和 text_model 呼叫 (保持不變) ---
        # (複製 modeling_smolvlm.py 的邏輯)
        # ... (呼叫 self.inputs_merger) ...
        # ... (呼叫 self.text_model) ...

        # --- return 邏輯 (保持不變) ---
        # ... (複製 modeling_smolvlm.py 的 return 邏輯) ...

        pass # Placeholder for brevity


# --- MyPS3VLMForConditionalGeneration (保持不變) ---
# (這個外殼類別只需要把 self.model 指向 MyPS3VLMModel 即可，
#  所以不需要修改 modeling_smolvlm_ps3.py 中定義的這個類別)
class MyPS3VLMForConditionalGeneration(Idefics3ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = MyPS3VLMModel(config) # 指向我們上面修改的核心
        # ... (lm_head 和 post_init 保持不變) ...
import torch

from src.model.seam import Seam


# SEAM with static tensor shapes for ONNX export (MVA edge deployment)
# Requires a fixed number of agent and lane tokens (see Av2DatasetFixed). Parameters are identical to Seam.
class SeamOnnx(Seam):
    def forward(self, data):
        # token budget is static, so agent / lane split can be done by slicing
        self.num_agent_tokens = data['x_centers'].shape[1]
        return super().forward(data)


    def _encode_agents(self, hist_feat, hist_valid_mask, hist_key_valid_mask):
        # encode all tracks instead of selecting the valid ones
        B, N, L, D = hist_feat.shape
        actor_feat = hist_feat.view(B * N, L, D)
        ts = torch.arange(L, device=actor_feat.device).view(1, -1, 1).repeat(B * N, 1, 1).float()
        actor_feat = torch.cat([actor_feat, ts], dim=-1)

        actor_feat = self.h_proj(actor_feat)
        kpm = (~hist_valid_mask).view(B * N, -1)
        for blk in self.h_embed:
            actor_feat = blk(actor_feat, key_padding_mask=kpm)
        actor_feat = torch.max(actor_feat, axis=1).values.view(B, N, -1)

        # fully masked (padded) tracks yield NaNs, invalid agents are zero as in Seam
        actor_feat = torch.nan_to_num(actor_feat)
        return torch.where(hist_key_valid_mask.unsqueeze(-1), actor_feat, torch.zeros_like(actor_feat))


    def _encode_target_context(self, target_encoder, target_mask, memory_new_y_hat):
        # keep all tokens per target region instead of compressing them to the valid ones
        B, K = target_encoder.shape[:2]
        target_encoder = target_encoder.reshape(B * K, -1, self.embed_dim)

        # fixed w.r.t. the Jetson implementation of the MVA paper (deviation in metrics <= 0.003):
        # boolean key padding mask (a float 0/1 mask adds +1 to masked scores instead of ignoring them)
        for blk in self.target_blocks:
            target_encoder = blk(target_encoder, key_padding_mask=target_mask)
        target_encoder = self.target_norm(target_encoder)

        # fixed w.r.t. the Jetson implementation of the MVA paper: embed the last predicted step as in Seam (not target_offset)
        target_center_embed = self.target_center_embed(memory_new_y_hat[:, :, -1].detach())
        target_encoder = target_encoder.view(B, K, -1, self.embed_dim) + target_center_embed.unsqueeze(2)
        return target_encoder, target_encoder, target_mask


    def _split_tokens(self, x, type_mask):
        # agent tokens come first, memory has the same token budget
        return x[:, :self.num_agent_tokens], x[:, self.num_agent_tokens:]

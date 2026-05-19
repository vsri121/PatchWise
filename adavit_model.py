import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
# PATCH EMBEDDING
# =====================================================

class PatchEmbedding(nn.Module):

    def __init__(self, image_size=32, patch_size=4, dim=256):

        super().__init__()

        self.patch_size = patch_size

        self.num_patches = (image_size // patch_size) ** 2

        patch_dim = 3 * patch_size * patch_size

        self.proj = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, img):

        p = self.patch_size

        x = img.unfold(2, p, p).unfold(3, p, p)

        x = x.contiguous().permute(0, 2, 3, 1, 4, 5)

        x = x.reshape(x.size(0), -1, 3 * p * p)

        return self.proj(x)


# =====================================================
# EARLY BLOCK WITH ATTENTION
# =====================================================

class EarlyBlockWithAttn(nn.Module):

    def __init__(self, dim, heads, mlp_dim, num_layers=2):

        super().__init__()

        layers = []

        for _ in range(num_layers):

            layers.append(
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=heads,
                    dim_feedforward=mlp_dim,
                    dropout=0.1,
                    batch_first=True,
                    norm_first=True,
                )
            )

        self.layers = nn.ModuleList(layers)

        self._last_attn_weights = None

        self._register_last_layer_hook()

    def _register_last_layer_hook(self):

        last_mha = self.layers[-1].self_attn

        def hook(module, input, output):

            if isinstance(output, tuple) and len(output) == 2:

                self._last_attn_weights = output[1]

        last_mha.register_forward_hook(hook)

    def forward(self, x):

        for layer in self.layers[:-1]:

            x = layer(x)

        last_layer = self.layers[-1]

        last_mha = last_layer.self_attn

        orig_forward = last_mha.forward

        self._last_attn_weights = None

        def forward_with_weights(query, key, value, **kwargs):

            kwargs["need_weights"] = True

            kwargs["average_attn_weights"] = True

            return orig_forward(query, key, value, **kwargs)

        last_mha.forward = forward_with_weights

        x = last_layer(x)

        last_mha.forward = orig_forward

        return x

    def get_patch_importance(self, num_patches):

        w = self._last_attn_weights

        if w is None:

            return None

        patch_w = w[:, 1:, 1:]

        importance = patch_w.sum(dim=1)

        importance = importance / (
            importance.sum(dim=1, keepdim=True) + 1e-8
        )

        return importance.detach()


# =====================================================
# A3C PATCH POLICY
# =====================================================

class A3CPatchPolicy(nn.Module):

    def __init__(self, dim, num_patches):

        super().__init__()

        self.num_patches = num_patches

        self.context_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        self.context_norm = nn.LayerNorm(dim)

        self.actor = nn.Sequential(
            nn.Linear(dim + 1, dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, 2),
        )

        self.critic = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

        nn.init.zeros_(self.critic[-1].weight)
        nn.init.zeros_(self.critic[-1].bias)

    def forward(
        self,
        patch_features,
        patch_importance,
        is_training=True,
        policy_active=True
    ):

        B, N, D = patch_features.shape

        ctx = self.context_norm(patch_features)

        ctx_out, _ = self.context_attn(
            ctx,
            ctx,
            ctx,
            need_weights=False
        )

        ctx = ctx + ctx_out

        if patch_importance is not None:

            imp = patch_importance.detach()

        else:

            imp = torch.ones(
                B,
                N,
                device=patch_features.device
            ) / N

        importance_feat = imp.unsqueeze(-1)

        actor_input = torch.cat(
            [ctx, importance_feat],
            dim=-1
        )

        actor_logits = self.actor(actor_input)

        actor_probs = F.softmax(actor_logits, dim=-1)

        entropy = -(
            actor_probs * (actor_probs + 1e-8).log()
        ).sum(dim=-1)

        if not policy_active:

            mask = torch.ones(
                B,
                N,
                device=patch_features.device
            )

            log_probs = F.log_softmax(
                actor_logits,
                dim=-1
            )[:, :, 1]

        elif is_training:

            dist = torch.distributions.Categorical(
                probs=actor_probs
            )

            action = dist.sample()

            log_probs = dist.log_prob(action)

            mask = action.float()

        else:

            mask = (
                actor_logits[:, :, 1]
                > actor_logits[:, :, 0]
            ).float()

            log_probs = F.log_softmax(
                actor_logits,
                dim=-1
            )[:, :, 1]

        state_repr = ctx.mean(dim=1)

        value = self.critic(state_repr)

        return (
            mask,
            log_probs,
            entropy,
            value,
            actor_logits,
            imp
        )


# =====================================================
# ADAVIT DYNAMIC
# =====================================================

class AdaViTDynamic(nn.Module):

    def __init__(
        self,
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=256,
        depth=8,
        heads=8,
        mlp_dim=512
    ):

        super().__init__()

        self.num_patches = (
            image_size // patch_size
        ) ** 2

        self.dim = dim

        self.patch_size = patch_size

        self.patch_embed = PatchEmbedding(
            image_size,
            patch_size,
            dim
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(
                1,
                self.num_patches + 1,
                dim
            ) * 0.02
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, dim) * 0.02
        )

        self.early_block = EarlyBlockWithAttn(
            dim,
            heads,
            mlp_dim,
            num_layers=2
        )

        self.policy = A3CPatchPolicy(
            dim,
            self.num_patches
        )

        main_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            main_layer,
            num_layers=max(depth - 2, 1)
        )

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(0.1),
            nn.Linear(dim, num_classes),
        )

    def forward(
        self,
        img,
        is_training=False,
        policy_active=True
    ):

        B = img.shape[0]

        patches = self.patch_embed(img)

        cls = self.cls_token.expand(B, -1, -1)

        x = torch.cat([cls, patches], dim=1)

        x = x + self.pos_embedding[:, :x.size(1)]

        x = self.early_block(x)

        patch_importance = self.early_block.get_patch_importance(
            self.num_patches
        )

        patch_features = x[:, 1:]

        cls_out = x[:, 0:1]

        (
            mask,
            log_probs,
            entropy,
            value,
            actor_logits,
            imp_used
        ) = self.policy(
            patch_features,
            patch_importance=patch_importance,
            is_training=is_training,
            policy_active=policy_active,
        )

        kept_list = []

        max_kept = 0

        for b in range(B):

            kept_idx = mask[b].nonzero(as_tuple=True)[0]

            if len(kept_idx) == 0:

                kept_idx = torch.arange(
                    min(4, self.num_patches),
                    device=img.device
                )

            kept_list.append(
                patch_features[b, kept_idx]
            )

            max_kept = max(
                max_kept,
                len(kept_idx)
            )

        padded = torch.zeros(
            B,
            max_kept,
            self.dim,
            device=img.device
        )

        pad_mask = torch.ones(
            B,
            max_kept + 1,
            dtype=torch.bool,
            device=img.device
        )

        pad_mask[:, 0] = False

        for b, k in enumerate(kept_list):

            length = k.size(0)

            padded[b, :length] = k

            pad_mask[b, 1:length + 1] = False

        x_main = torch.cat(
            [cls_out, padded],
            dim=1
        )

        x_main = self.transformer(
            x_main,
            src_key_padding_mask=pad_mask
        )

        logits = self.head(x_main[:, 0])

        return {
            "logits": logits,
            "mask": mask,
            "log_probs": log_probs,
            "entropy": entropy,
            "value": value,
            "actor_logits": actor_logits,
            "keep_prob": F.softmax(
                actor_logits,
                dim=-1
            )[:, :, 1].detach(),
            "patch_importance": imp_used,
        }
# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Style encoder of GST-Tacotron."""

from typing import Sequence

import math

import torch
import torch.nn as nn


class BaseMultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer .
        using named as BaseMultiHeadedAttention.
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(BaseMultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)

class MultiHeadedAttention(BaseMultiHeadedAttention):
    """Multi head attention module with different input dimension."""

    def __init__(self, q_dim, k_dim, v_dim, n_head, n_feat, dropout_rate=0.0):
        """Initialize multi head attention module."""
        # NOTE(kan-bayashi): Do not use super().__init__() here since we want to
        #   overwrite BaseMultiHeadedAttention.__init__() method.
        torch.nn.Module.__init__(self)
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = torch.nn.Linear(q_dim, n_feat)
        self.linear_k = torch.nn.Linear(k_dim, n_feat)
        self.linear_v = torch.nn.Linear(v_dim, n_feat)
        self.linear_out = torch.nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout_rate)

class StyleEncoder(torch.nn.Module):
    """Style encoder.

    This module is style encoder introduced in `Style Tokens: Unsupervised Style
    Modeling, Control and Transfer in End-to-End Speech Synthesis`.

    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
        Speech Synthesis`: https://arxiv.org/abs/1803.09017

    Args:
        idim (int, optional): Dimension of the input mel-spectrogram.
        gst_tokens (int, optional): The number of GST embeddings.
        gst_token_dim (int, optional): Dimension of each GST embedding.
        gst_heads (int, optional): The number of heads in GST multihead attention.
        conv_layers (int, optional): The number of conv layers in the reference encoder.
        conv_chans_list: (Sequence[int], optional):
            List of the number of channels of conv layers in the referece encoder.
        conv_kernel_size (int, optional):
            Kernel size of conv layers in the reference encoder.
        conv_stride (int, optional):
            Stride size of conv layers in the reference encoder.
        gru_layers (int, optional): The number of GRU layers in the reference encoder.
        gru_units (int, optional): The number of GRU units in the reference encoder.

    Todo:
        * Support manual weight specification in inference.

    """

    def __init__(
        self,
        idim: int = 80,
        gst_tokens: int = 10,
        gst_token_dim: int = 256,
        gst_heads: int = 4,
        conv_layers: int = 6,
        conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        gru_layers: int = 1,
        gru_units: int = 128,
    ):
        """Initilize global style encoder module."""
        super(StyleEncoder, self).__init__()

        self.ref_enc = ReferenceEncoder(
            idim=idim,
            conv_layers=conv_layers,
            conv_chans_list=conv_chans_list,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            gru_layers=gru_layers,
            gru_units=gru_units,
        )
        self.stl = StyleTokenLayer(
            ref_embed_dim=gru_units,
            gst_tokens=gst_tokens,
            gst_token_dim=gst_token_dim,
            gst_heads=gst_heads,
        )

    def forward(self, speech: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            speech (Tensor): Batch of padded target features (B, Lmax, odim).

        Returns:
            Tensor: Style token embeddings (B, token_dim).

        """
        ref_embs = self.ref_enc(speech)
        style_embs = self.stl(ref_embs)

        return style_embs


class ReferenceEncoder(torch.nn.Module):
    """Reference encoder module.

    This module is reference encoder introduced in `Style Tokens: Unsupervised Style
    Modeling, Control and Transfer in End-to-End Speech Synthesis`.

    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
        Speech Synthesis`: https://arxiv.org/abs/1803.09017

    Args:
        idim (int, optional): Dimension of the input mel-spectrogram.
        conv_layers (int, optional): The number of conv layers in the reference encoder.
        conv_chans_list: (Sequence[int], optional):
            List of the number of channels of conv layers in the referece encoder.
        conv_kernel_size (int, optional):
            Kernel size of conv layers in the reference encoder.
        conv_stride (int, optional):
            Stride size of conv layers in the reference encoder.
        gru_layers (int, optional): The number of GRU layers in the reference encoder.
        gru_units (int, optional): The number of GRU units in the reference encoder.

    """

    def __init__(
        self,
        idim=80,
        conv_layers: int = 6,
        conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        gru_layers: int = 1,
        gru_units: int = 128,
    ):
        """Initilize reference encoder module."""
        super(ReferenceEncoder, self).__init__()

        # check hyperparameters are valid
        assert conv_kernel_size % 2 == 1, "kernel size must be odd."
        assert (
            len(conv_chans_list) == conv_layers
        ), "the number of conv layers and length of channels list must be the same."

        convs = []
        padding = (conv_kernel_size - 1) // 2
        for i in range(conv_layers):
            conv_in_chans = 1 if i == 0 else conv_chans_list[i - 1]
            conv_out_chans = conv_chans_list[i]
            convs += [
                torch.nn.Conv2d(
                    conv_in_chans,
                    conv_out_chans,
                    kernel_size=conv_kernel_size,
                    stride=conv_stride,
                    padding=padding,
                    # Do not use bias due to the following batch norm
                    bias=False,
                ),
                torch.nn.BatchNorm2d(conv_out_chans),
                torch.nn.ReLU(inplace=True),
            ]
        self.convs = torch.nn.Sequential(*convs)

        self.conv_layers = conv_layers
        self.kernel_size = conv_kernel_size
        self.stride = conv_stride
        self.padding = padding

        # get the number of GRU input units
        gru_in_units = idim
        for i in range(conv_layers):
            gru_in_units = (
                gru_in_units - conv_kernel_size + 2 * padding
            ) // conv_stride + 1
        gru_in_units *= conv_out_chans
        self.gru = torch.nn.GRU(gru_in_units, gru_units, gru_layers, batch_first=True)

    def forward(self, speech: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            speech (Tensor): Batch of padded target features (B, Lmax, idim).

        Returns:
            Tensor: Reference embedding (B, gru_units)

        """
        batch_size = speech.size(0)
        xs = speech.unsqueeze(1)  # (B, 1, Lmax, idim)
        hs = self.convs(xs).transpose(1, 2)  # (B, Lmax', conv_out_chans, idim')
        # NOTE(kan-bayashi): We need to care the length?
        time_length = hs.size(1)
        hs = hs.contiguous().view(batch_size, time_length, -1)  # (B, Lmax', gru_units)
        self.gru.flatten_parameters()
        _, ref_embs = self.gru(hs)  # (gru_layers, batch_size, gru_units)
        ref_embs = ref_embs[-1]  # (batch_size, gru_units)

        return ref_embs


class StyleTokenLayer(torch.nn.Module):
    """Style token layer module.

    This module is style token layer introduced in `Style Tokens: Unsupervised Style
    Modeling, Control and Transfer in End-to-End Speech Synthesis`.

    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
        Speech Synthesis`: https://arxiv.org/abs/1803.09017

    Args:
        ref_embed_dim (int, optional): Dimension of the input reference embedding.
        gst_tokens (int, optional): The number of GST embeddings.
        gst_token_dim (int, optional): Dimension of each GST embedding.
        gst_heads (int, optional): The number of heads in GST multihead attention.
        dropout_rate (float, optional): Dropout rate in multi-head attention.

    """

    def __init__(
        self,
        ref_embed_dim: int = 128,
        gst_tokens: int = 10,
        gst_token_dim: int = 256,
        gst_heads: int = 4,
        dropout_rate: float = 0.0,
    ):
        """Initilize style token layer module."""
        super(StyleTokenLayer, self).__init__()

        gst_embs = torch.randn(gst_tokens, gst_token_dim // gst_heads)
        self.register_parameter("gst_embs", torch.nn.Parameter(gst_embs))
        self.mha = MultiHeadedAttention(
            q_dim=ref_embed_dim,
            k_dim=gst_token_dim // gst_heads,
            v_dim=gst_token_dim // gst_heads,
            n_head=gst_heads,
            n_feat=gst_token_dim,
            dropout_rate=dropout_rate,
        )

    def forward(self, ref_embs: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            ref_embs (Tensor): Reference embeddings (B, ref_embed_dim).

        Returns:
            Tensor: Style token embeddings (B, gst_token_dim).

        """
        batch_size = ref_embs.size(0)
        # (num_tokens, token_dim) -> (batch_size, num_tokens, token_dim)
        gst_embs = torch.tanh(self.gst_embs).unsqueeze(0).expand(batch_size, -1, -1)
        # NOTE(kan-bayashi): Shoule we apply Tanh?
        ref_embs = ref_embs.unsqueeze(1)  # (batch_size, 1 ,ref_embed_dim)
        style_embs = self.mha(ref_embs, gst_embs, gst_embs, None)

        return style_embs.squeeze(1)



if __name__ == "__main__":
    model = StyleEncoder(gst_tokens=4)
    device = torch.device("cuda:0")
    model = model.to(device)
    input = torch.randn(2, 80, 100).to(device)
    output = model(input)
    print(output.shape)
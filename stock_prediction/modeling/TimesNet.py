import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import numpy as np
from torch import Tensor
from einops import *

from layers.Formers.Embed import DataEmbedding
from layers.TimesNet.Conv_Blocks import Inception_Block_V1
from utils.ExpConfigs import ExpConfigs
from loguru import logger

class Model(nn.Module):
    """
    - paper: "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis" (ICLR 2023)
    - paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    - code adapted from: https://github.com/thuml/Time-Series-Library
    """

    def __init__(self, configs: ExpConfigs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.configs = configs
        self.seq_len = configs.seq_len_max_irr or configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if configs.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        elif self.task_name == ['imputation', 'anomaly_detection']:
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        elif self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.n_classes)
        else:
            raise NotImplementedError

    def forward(
        self, 
        x: Tensor,
        x_mark: Tensor = None, 
        y: Tensor = None,
        y_mask: Tensor = None,
        y_class: Tensor = None,
        **kwargs
    ):
        # BEGIN adaptor
        BATCH_SIZE, SEQ_LEN, ENC_IN = x.shape
        Y_LEN = self.pred_len
        if x_mark is None:
            x_mark = repeat(torch.arange(end=x.shape[1], dtype=x.dtype, device=x.device) / x.shape[1], "L -> B L 1", B=x.shape[0])
        if y is None:
            if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
                logger.warning(f"y is missing for the model input. This is only reasonable when the model is testing flops!")
            y = torch.ones((BATCH_SIZE, Y_LEN, ENC_IN), dtype=x.dtype, device=x.device)
        if y_mask is None:
            y_mask = torch.ones_like(y, device=y.device, dtype=y.dtype)
        if y_class is None:
            if self.configs.task_name == "classification":
                logger.warning(f"y_class is missing for the model input. This is only reasonable when the model is testing flops!")
            y_class = torch.ones((BATCH_SIZE), dtype=x.dtype, device=x.device)

        x_mark[x_mark == 1] = 0.9999
        # END adaptor

        if self.configs.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x, x_mark)
            f_dim = -1 if self.configs.features == 'MS' else 0
            PRED_LEN = y.shape[1]
            return {
                "pred": dec_out[:, -PRED_LEN:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:]
            }
        elif self.configs.task_name == "classification":
            return {
                "pred_class": self.classification(x, x_mark),
                "true_class": y_class
            }
        else:
            raise NotImplementedError

    def forecast(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        
        # Handle x_mark_enc properly - it should be a mask for padding
        # If x_mark_enc is not a proper mask, create one
        if x_mark_enc.shape[-1] == 1:  # Time features
            # Create a mask that's 1 for all positions (no padding)
            mask = torch.ones_like(output, device=output.device)
        else:
            mask = x_mark_enc
            
        # zero-out padding embeddings
        output = output * mask
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    
    # Ensure top_list values are valid (not 0 and not too large)
    top_list = np.maximum(top_list, 1)  # Ensure minimum value is 1
    top_list = np.minimum(top_list, x.shape[1] // 2)  # Ensure maximum value is reasonable
    
    period = x.shape[1] // top_list
    # Ensure period values are valid (not 0 and not too large)
    period = np.maximum(period, 1)  # Ensure minimum period is 1
    period = np.minimum(period, x.shape[1])  # Ensure maximum period is sequence length
    
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs: ExpConfigs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len_max_irr or configs.seq_len # equal to seq_len_max_irr if not None, else seq_len
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        self.k = 5
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=6),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=6)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            
            # Additional safety check for period
            if period <= 0 or period > T:
                # Use a default period if the calculated one is invalid
                period = max(1, T // 4)  # Default to quarter of sequence length
            
            # For classification, we only use the actual sequence length
            # For forecasting, we use seq_len + pred_len
            target_length = T  # Use actual input length for classification
            
            # padding
            if target_length % period != 0:
                length = ((target_length // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - target_length), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = target_length
                out = x
            
            # Additional safety check for reshape dimensions
            if length // period <= 0 or period <= 0:
                # Skip this period if dimensions are invalid
                res.append(x)  # Use original input as fallback
                continue
                
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :target_length, :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res
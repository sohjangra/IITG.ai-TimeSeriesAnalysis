"""
model_architecture.py

Model definition for the volatility-forecasting half of the pipeline.

`VolatilityModel` is a hybrid PatchTST(+LoRA) -> LSTM -> MLP regression head
that consumes a (batch, seq_length, num_features) window of engineered
alpha features and predicts a single scalar: next-step trailing realized
volatility.

The direction/return model (previously a neural `ReturnModel`) has been
REMOVED from this pipeline entirely -- direction/return prediction is now
handled by an XGBoost Booster trained directly in the training script with
a custom focal-loss objective, since gradient-boosted trees do not need a
PyTorch architecture module.

Requires:
    pip install --upgrade torchao peft transformers
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import PatchTSTModel, PatchTSTConfig


class VolatilityModel(nn.Module):
    """
    LSTM + PatchTST(LoRA) + dense head, dedicated to volatility regression.

    Architecture
    ------------
    1. PatchTST backbone (frozen base weights + LoRA adapters on the
       query/value projections): treats each of `num_features` channels as
       an independent time series, splits the `seq_length`-step window into
       patches of length 2, and embeds each patch into a `d_model`-dim
       vector via self-attention.
    2. Reshape: the (batch, channels, patches, d_model) PatchTST output is
       permuted and flattened per-patch across channels, producing a
       (batch, patches, channels * d_model) sequence.
    3. LSTM: compresses that per-patch sequence into a single hidden-state
       vector (sequence-to-vector encoding).
    4. MLP regression head: maps the LSTM's final hidden state down to a
       single scalar volatility prediction.

    Intended to be trained with `exponential_asymmetric_loss` (see the
    training script), which penalizes under-estimating volatility more
    heavily than over-estimating it.
    """

    def __init__(self, num_features, seq_length=10, lora_rank=8, lstm_hidden=64):
        super().__init__()

        config = PatchTSTConfig(
            num_input_channels=num_features,
            context_length=seq_length,
            patch_length=2,
            d_model=128,
        )
        self.base_transformer = PatchTSTModel(config)

        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=32, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1, bias="none",
        )
        self.transformer_with_lora = get_peft_model(self.base_transformer, lora_config)

        flattened_input_size = num_features * 128
        self.lstm = nn.LSTM(
            input_size=flattened_input_size,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.lstm_dropout = nn.Dropout(0.2)

        self.vol_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_length, num_features)

        Returns
        -------
        torch.Tensor, shape (batch,)
            Predicted next-step trailing volatility, one scalar per sample.
        """
        outputs = self.transformer_with_lora(past_values=x)
        b, c, p, d = outputs.last_hidden_state.shape

        sequential_states = outputs.last_hidden_state.permute(0, 2, 1, 3).reshape(b, p, c * d)
        lstm_out, (h_n, c_n) = self.lstm(sequential_states)

        recurrent_feature = self.lstm_dropout(h_n.squeeze(0))

        vol_out = self.vol_head(recurrent_feature).squeeze(-1)
        return vol_out


class VolWrapper(nn.Module):
    """
    Thin wrapper around `VolatilityModel` so SHAP's GradientExplainer sees
    a single scalar output per sample with an explicit trailing dimension
    (shape [batch, 1]) rather than a bare (batch,) tensor.
    """

    def __init__(self, model: VolatilityModel):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).unsqueeze(-1)

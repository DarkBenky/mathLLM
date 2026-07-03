import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from defines import VOCAB, INPUT_SIZE


class TransformerBlock(layers.Layer):

    def __init__(self, d_model: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
        )

        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(d_model),
        ])

        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.do1 = layers.Dropout(dropout)
        self.do2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        a = self.attn(x, x, training=training)
        a = self.do1(a, training=training)
        x = self.ln1(x + a)

        f = self.ffn(x)
        f = self.do2(f, training=training)
        x = self.ln2(x + f)

        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate,
        })
        return cfg


class MathTransformer(keras.Model):

    def __init__(self,
                 d_model: int = 192,
                 num_heads: int = 6,
                 num_layers: int = 4,
                 ff_dim: int = 768,
                 dropout: float = 0.1,
                 vocab_size: int | None = None,
                 input_size: int | None = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        self.vocab_size = vocab_size or len(VOCAB)
        self.input_size = input_size or INPUT_SIZE

        self.input_proj = layers.Dense(d_model, name="input_proj")

        self.pos_enc = self.add_weight(
            name="positional_encoding",
            shape=(1, self.input_size, d_model),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )

        self.blocks = [
            TransformerBlock(d_model, num_heads, ff_dim, dropout,
                             name=f"transformer_block_{i}")
            for i in range(num_layers)
        ]

        self.token_head = layers.Dense(self.vocab_size, name="token_head")
        self.feedback_norm = layers.LayerNormalization(epsilon=1e-6, name="feedback_norm")

        self.value_pool = layers.GlobalAveragePooling1D(name="value_pool")
        self.value_dense = layers.Dense(64, activation="gelu", name="value_dense")
        self.value_do = layers.Dropout(dropout, name="value_dropout")
        self.value_out = layers.Dense(1, name="value_out")

    def call(self, inputs, training=False):
        x = self.input_proj(inputs)
        x = x + self.pos_enc[:, :self.input_size, :]

        for block in self.blocks:
            x = block(x, training=training)

        token_logits = self.token_head(x)

        v = self.value_pool(x)
        v = self.value_dense(v)
        v = self.value_do(v, training=training)
        value_pred = self.value_out(v)

        return token_logits, value_pred

    def build_graph(self):
        batch_size = 2
        dummy = tf.zeros((batch_size, self.input_size, self.vocab_size))
        self(dummy)

        total = sum(tf.size(w).numpy() for w in self.trainable_weights)
        non_trainable = sum(tf.size(w).numpy() for w in self.non_trainable_weights)

        print(f"\n{'='*60}")
        print(f"  MathTransformer — Parameter Summary")
        print(f"{'='*60}")
        print(f"  d_model:      {self.d_model}")
        print(f"  num_heads:    {self.num_heads}")
        print(f"  num_layers:   {self.num_layers}")
        print(f"  ff_dim:       {self.ff_dim}")
        print(f"  vocab_size:   {self.vocab_size}")
        print(f"  input_size:   {self.input_size}")
        print(f"  dropout:      {self.dropout_rate}")
        print(f"{'='*60}")
        print(f"  Trainable params:      {total:>12,}")
        print(f"  Non-trainable params:  {non_trainable:>12,}")
        print(f"  Total params:          {total + non_trainable:>12,}")
        print(f"{'='*60}\n")
        return total

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate,
            "vocab_size": self.vocab_size,
            "input_size": self.input_size,
        })
        return cfg


if __name__ == "__main__":
    model = MathTransformer()
    total_params = model.build_graph()

    batch = tf.zeros((4, INPUT_SIZE, len(VOCAB)))
    token_logits, value_pred = model(batch, training=True)
    print(f"Token logits shape:  {token_logits.shape}")
    print(f"Value pred shape:    {value_pred.shape}")

    token_probs = tf.nn.softmax(token_logits, axis=-1)
    pos_sums = tf.reduce_sum(token_probs, axis=-1)
    print(f"Per-position sum range: [{tf.reduce_min(pos_sums):.4f}, "
          f"{tf.reduce_max(pos_sums):.4f}]  (should be ~1)")


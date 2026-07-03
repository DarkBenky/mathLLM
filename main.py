from model import MathTransformer
from generateData import tokenizeExpression, generateExpression
from defines import VOCAB, INPUT_SIZE, OPERATORS, ITERATION
import wandb
import numpy as np
import tensorflow as tf

EPOCHS = 1_000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
VIZ_INTERVAL = 50


def decodeTokens(probs):
    indices = np.argmax(probs, axis=-1)
    chars = []
    for idx in indices:
        char = VOCAB[idx]
        if char == "EMPTY":
            break
        chars.append(char)
    return "".join(chars)


if __name__ == "__main__":
    wandb.init(project="mathLLM")

    model = MathTransformer()
    total_params = model.build_graph()

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)

    for epoch in range(EPOCHS):
        expressions = []
        results = []
        for _ in range(BATCH_SIZE):
            exp, res = generateExpression()
            expressions.append(exp)
            results.append(res)

        X_np = np.array([tokenizeExpression(exp) for exp in expressions])
        y_tok_np = np.array([tokenizeExpression(str(int(round(res)))) for res in results])
        y_val_np = np.array(results, dtype=np.float32).reshape(-1, 1)

        X = tf.convert_to_tensor(X_np, dtype=tf.float32)
        y_tok = tf.convert_to_tensor(y_tok_np, dtype=tf.float32)
        y_val = tf.convert_to_tensor(y_val_np, dtype=tf.float32)

        empty_idx = VOCAB.index("EMPTY")
        tok_weights = 1.0 + 9.0 * (1.0 - y_tok[:, :, empty_idx])

        with tf.GradientTape() as tape:
            token_logits, value_pred = model(X, training=True)

            ce_per_pos = tf.keras.losses.categorical_crossentropy(
                y_tok, token_logits, from_logits=True)
            token_loss = tf.reduce_mean(ce_per_pos * tok_weights)

            value_loss = tf.reduce_mean(tf.square(
                (value_pred - y_val) / (tf.abs(y_val) + 1.0)))
            loss = token_loss + value_loss

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        abs_error = tf.reduce_mean(tf.abs(value_pred - y_val))

        wandb.log({
            "epoch": epoch,
            "loss": loss.numpy(),
            "token_loss": token_loss.numpy(),
            "value_loss": value_loss.numpy(),
            "mae": abs_error.numpy(),
        })

        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | loss: {loss.numpy():.4f} | "
                  f"tok: {token_loss.numpy():.4f} | val: {value_loss.numpy():.4f} | "
                  f"MAE: {abs_error.numpy():.1f}")

        if epoch % VIZ_INTERVAL == 0:
            X_viz = tf.convert_to_tensor(X_np[:1], dtype=tf.float32)

            print(f"\n{'='*50}")
            print(f"Epoch {epoch} — True: {expressions[0]} = {results[0]}")
            print(f"{'='*50}")
            print(f"{'iter':<6} {'decoded':<30} {'value_pred':>10}")
            print("-" * 50)

            decoded = decodeTokens(X_np[0])
            print(f"{0:<6} {decoded:<30} {'—':>10}")

            for i in range(ITERATION):
                token_logits, value_pred = model(X_viz, training=False)
                probs = tf.nn.softmax(token_logits, axis=-1).numpy()[0]
                decoded = decodeTokens(probs)
                print(f"{i+1:<6} {decoded:<30} {value_pred.numpy()[0,0]:>10.1f}")
                X_viz = probs[np.newaxis, ...]

            print(f"{'='*50}\n")
        
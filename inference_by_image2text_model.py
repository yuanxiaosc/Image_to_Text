import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PIL_Image
import os
from dataset_utils import get_text_tokenizer, image_text_tfrecored_2_dataset
from image2text_model import CNN_Encoder, RNN_Decoder


def plot_attention(idx, image, result, attention_plot, out_attention_weight_text_image_path, plt_show=False):
    temp_image = image[0].numpy() / 255.0

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    if not (len_result % 2) == 0:
        len_result = len_result - 1

    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.savefig(f"{os.path.join(out_attention_weight_text_image_path, str(idx))}.png")
    if plt_show:
        plt.show()


def restore_model(checkpoint_path, vocab_size):
    encoder = CNN_Encoder()
    decoder = RNN_Decoder(vocab_size)
    optimizer = tf.keras.optimizers.Adam()

    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!')

    return encoder, decoder,

def load_image(image_path, new_height=299, new_width=299):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (new_height, new_width))
    return img

def main(model_predicte_number, checkpoint_path, image_TFRecord_path, text_TFRecord_path,
         text_max_length, attention_features_shape, plot_image_attention, out_attention_weight_text_image_path):
    def evaluate(image):
        attention_plot = np.zeros((text_max_length, attention_features_shape))

        hidden = decoder.reset_state(batch_size=1)

        image_features_encoder = encoder(image)

        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        result = []
        for i in range(text_max_length):
            predictions, hidden, attention_weights = decoder(dec_input, image_features_encoder, hidden)

            attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()
            result.append(tokenizer.index_word[predicted_id])

            if tokenizer.index_word[predicted_id] == '<end>':
                return result, attention_plot

            dec_input = tf.expand_dims([predicted_id], 0)

        attention_plot = attention_plot[:len(result), :]
        return result, attention_plot

    # restore tokenizer
    tokenizer = get_text_tokenizer(text_tokenizer_path="text_tokenizer")
    vocab_size = len(tokenizer.word_index) + 1

    # Preparing validation set data
    dataset = image_text_tfrecored_2_dataset(image_TFRecord_path, text_TFRecord_path, 1, None)

    # restore image caption model
    encoder, decoder = restore_model(checkpoint_path, vocab_size)

    # model prediction
    for (number, (image, text)) in enumerate(dataset.take(model_predicte_number)):
        target_list = text[0].numpy().tolist()
        predict_caption = ' '.join([tokenizer.index_word[i] for i in target_list if i not in [0]])

        result, attention_plot = evaluate(image)

        print('predicte_caption:', predict_caption)
        print('Prediction Caption:', ' '.join(result))
        if plot_image_attention:
            PIL_Image.open(image)
        if not os.path.exists(out_attention_weight_text_image_path):
            os.mkdir(out_attention_weight_text_image_path)
        plot_attention(number, image, result, attention_plot, out_attention_weight_text_image_path, plt_show=False)
        print("")


if __name__ == "__main__":
    model_predicte_number = 10
    text_max_length = 30
    checkpoint_path = "checkpoints/train"
    image_TFRecord_path = "image_text_TFRecord/val_image_path.tfre"
    text_TFRecord_path = "image_text_TFRecord/val_text_content.tfre"
    attention_features_shape = 64
    plot_image_attention = False
    out_attention_weight_text_image_path = "output_inference_image"

    main(model_predicte_number, checkpoint_path, image_TFRecord_path, text_TFRecord_path,
         text_max_length, attention_features_shape, plot_image_attention, out_attention_weight_text_image_path)

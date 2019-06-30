import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time

from image2text_model import CNN_Encoder, RNN_Decoder, loss_function
from dataset_utils import get_text_tokenizer, image_text_tfrecored_2_dataset, image_text_tfrecored_2_artificial_dataset


def plot_loss_picture(loss_plot, plt_show=False):
    plt.plot(loss_plot)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.savefig("loss.png")
    if plt_show:
        plt.show()


def main(EPOCHS, BATCH_SIZE, NUMBER_STEP, BUFFER_SIZE, checkpoint_path,
         image_TFRecord_path, text_TFRecord_path, new_height=299, new_width=299):
    @tf.function
    def train_step(img_tensor, target):
        loss = 0
        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = decoder.reset_state(batch_size=target.shape[0])
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)
        with tf.GradientTape() as tape:
            features = encoder(img_tensor)
            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = decoder(dec_input, features, hidden)
                loss += loss_function(target[:, i], predictions)
                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)
        total_loss = (loss / int(target.shape[1]))
        trainable_variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss, total_loss

    # prepare dataset
    if not os.path.exists(image_TFRecord_path):
        dataset = image_text_tfrecored_2_artificial_dataset(BATCH_SIZE, BUFFER_SIZE)
    else:
        dataset = image_text_tfrecored_2_dataset(image_TFRecord_path, text_TFRecord_path,
                                                 BATCH_SIZE, BUFFER_SIZE,
                                                 new_height, new_width)

    # restore tokenizer
    tokenizer = get_text_tokenizer(text_tokenizer_path="text_tokenizer")
    vocab_size = len(tokenizer.word_index) + 1

    # create model
    encoder = CNN_Encoder()
    decoder = RNN_Decoder(vocab_size)
    optimizer = tf.keras.optimizers.Adam()

    # create checkpoint
    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!')

    # train
    loss_plot = []
    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset):
            batch_loss, t_loss = train_step(img_tensor, target)
            total_loss += t_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(
                    epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / NUMBER_STEP)

        if epoch % 5 == 0:
            ckpt_manager.save()

        print('Epoch {} Loss {:.6f}'.format(epoch + 1, total_loss / NUMBER_STEP))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    # print(f"loss plot {loss_plot}")
    plot_loss_picture(loss_plot, plt_show=False)


if __name__ == "__main__":
    # Feel free to change these parameters according to your system's configuration
    EPOCHS = 20
    BATCH_SIZE = 64
    BUFFER_SIZE = 1000
    TRAIN_NUMBER = 1000
    NUMBER_STEP = TRAIN_NUMBER // BATCH_SIZE
    checkpoint_path = "./checkpoints/train"
    image_TFRecord_path = "./image_text_TFRecord/train_image_path.tfre"
    text_TFRecord_path = "./image_text_TFRecord/train_text_content.tfre"

    main(EPOCHS, BATCH_SIZE, NUMBER_STEP, BUFFER_SIZE, checkpoint_path, image_TFRecord_path, text_TFRecord_path)

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle


def dowload_MS_COCO_raw_data(download_folder=None):
    """
    data from http://cocodataset.org/#home
    :return: annotation_file (json file path) ...captions_train2014.json
            img_file_dir (raw image file folder) ... train2014
    """
    if download_folder is not None and os.path.exists(download_folder):
        annotation_file = download_folder + '/annotations/captions_train2014.json'
        img_file_dir = download_folder + '/train2014'
        return annotation_file, img_file_dir

    annotation_zip = tf.keras.utils.get_file('captions.zip',
                                             cache_subdir=os.path.abspath('.'),
                                             origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                             extract=True)
    annotation_file = os.path.dirname(annotation_zip) + '/annotations/captions_train2014.json'

    name_of_zip = 'train2014.zip'
    if not os.path.exists(os.path.abspath('.') + '/' + name_of_zip):
        image_zip = tf.keras.utils.get_file(name_of_zip,
                                            cache_subdir=os.path.abspath('.'),
                                            origin='http://images.cocodataset.org/zips/train2014.zip',
                                            extract=True)
        img_file_dir = os.path.dirname(image_zip) + '/train2014/'
    else:
        img_file_dir = os.path.abspath('.') + '/train2014/'

    return annotation_file, img_file_dir


def read_raw_image_and_text_file(annotation_file, img_file_dir, num_examples=None):
    # Read the json file
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Store captions and image names in vectors
    all_captions = []
    all_img_name_vector = []

    for annot in annotations['annotations']:
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        full_coco_image_path = os.path.join(img_file_dir, 'COCO_train2014_' + '%012d.jpg' % (image_id))
        all_img_name_vector.append(full_coco_image_path)
        all_captions.append(caption)

    # Shuffle captions and image_names together
    # Set a random state
    train_captions, img_name_vector = shuffle(all_captions, all_img_name_vector, random_state=1)

    # Select the first num_examples captions from the shuffled set, None for all data
    train_captions = train_captions[:num_examples]
    img_name_vector = img_name_vector[:num_examples]

    print(f"train_captions numbers {len(train_captions)}\t img_name_vector numbers {len(img_name_vector)}")
    return train_captions, img_name_vector


def preprocess_and_tokenize_texts(train_texts, top_k):
    """
    :param train_texts: texts list
    :param top_k: limit the vocabulary size to the top k words (to save memory)
    :return: text_vector (token and pad)
    """

    # Find the maximum length of any text in our dataset
    def calc_max_length(tensor):
        return max(len(t) for t in tensor)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_texts)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    with open("text_tokenizer", "wb") as f:
        pickle.dump(tokenizer, f)

    # Create the tokenized vectors
    train_seqs = tokenizer.texts_to_sequences(train_texts)

    # Pad each vector to the max_length of the texts
    # If you do not provide a max_length value, pad_sequences calculates it automatically
    text_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    # Calculates the max_length, which is used to store the attention weights
    max_length = calc_max_length(train_seqs)
    print(f"max_length:\t{max_length}")

    return text_vector


def get_text_tokenizer(text_tokenizer_path="text_tokenizer"):
    with open(text_tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
        return tokenizer


def store_image_path_text_content_TFRecord(image_path_list, text_content_list, test_size=0.2,
                                           save_TFRecord_dir="image_text_TFRecord"):
    if not os.path.exists(save_TFRecord_dir):
        os.mkdir(save_TFRecord_dir)

    def store_image_path_tfrecord(image_path_list, save_dataset_path):
        image_ds = tf.data.Dataset.from_tensor_slices(image_path_list)
        image_ds = image_ds.map(tf.io.serialize_tensor)
        tfrec = tf.data.experimental.TFRecordWriter(save_dataset_path)
        tfrec.write(image_ds)

    def store_text_tfrecord(text_content_list, save_dataset_path):
        text_ds = tf.data.Dataset.from_tensor_slices(text_content_list)
        text_ds = text_ds.map(tf.io.serialize_tensor)
        tfrec = tf.data.experimental.TFRecordWriter(save_dataset_path)
        tfrec.write(text_ds)

    # Create training and validation sets using an 80-20 split
    image_path_list_train, image_path_list_val, text_content_list_train, text_content_list_val = train_test_split(
        image_path_list, text_content_list,
        test_size=test_size, random_state=0)
    train_image_path = os.path.join(save_TFRecord_dir, "train_image_path.tfre")
    val_image_path = os.path.join(save_TFRecord_dir, "val_image_path.tfre")
    train_text_path = os.path.join(save_TFRecord_dir, "train_text_content.tfre")
    val_text_path = os.path.join(save_TFRecord_dir, "val_text_content.tfre")

    store_image_path_tfrecord(image_path_list_train, train_image_path)
    store_image_path_tfrecord(image_path_list_val, val_image_path)
    store_text_tfrecord(text_content_list_train, train_text_path)
    store_text_tfrecord(text_content_list_val, val_text_path)


def image_text_tfrecored_2_dataset(image_TFRecord_path, text_TFRecord_path,
                                   BATCH_SIZE=64, BUFFER_SIZE=1000,
                                   new_height=299, new_width=299):
    def load_image(image_path):
        image_path = tf.io.parse_tensor(image_path, out_type=tf.string)
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (new_height, new_width))
        return img

    def parse_text(x):
        x = tf.io.parse_tensor(x, out_type=tf.int32)
        return x

    image_dataset = tf.data.TFRecordDataset(image_TFRecord_path)
    image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    text_dataset = tf.data.TFRecordDataset(text_TFRecord_path)
    text_dataset = text_dataset.map(parse_text, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    image_text_dataset = tf.data.Dataset.zip((image_dataset, text_dataset))
    # Shuffle and batch
    if BATCH_SIZE == 1:
        image_text_dataset = image_text_dataset.batch(BATCH_SIZE)
    else:
        image_text_dataset = image_text_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
        image_text_dataset = image_text_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return image_text_dataset


def image_text_tfrecored_2_artificial_dataset(BATCH_SIZE=64, BUFFER_SIZE=1000):
    image_dataset = tf.data.Dataset.from_tensor_slices(np.random.random(size=[1000, 299, 299, 3]).astype(np.float32))
    text_dataset = tf.data.Dataset.from_tensor_slices(np.random.randint(0, 1000, size=[1000, 31]).astype(np.int32))

    image_text_dataset = tf.data.Dataset.zip((image_dataset, text_dataset))
    # Shuffle and batch
    image_text_dataset = image_text_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    image_text_dataset = image_text_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return image_text_dataset


def check_tfrecord_data(image_TFRecord_path="image_text_TFRecord/train_image_path.tfre",
                        text_TFRecord_path="image_text_TFRecord/train_text_content.tfre",
                        text_tokenizer_path="text_tokenizer"):
    tokenizer = get_text_tokenizer(text_tokenizer_path)
    dataset = image_text_tfrecored_2_dataset(image_TFRecord_path, text_TFRecord_path, 3, 100, 299, 299)
    number = 1
    for batch_image, batch_text in dataset.take(1):
        for image, text in zip(batch_image, batch_text):
            target_list = text.numpy().tolist()
            target_text = ' '.join([tokenizer.index_word[i] for i in target_list if i not in [0]])
            print(f"image.shape {image.shape}")
            print(f"text.numpy() {text.numpy()}")
            print(f"target_text {target_text}")
            plt.title(target_text)
            plt.imshow(image.numpy() / 255)
            plt.savefig(f"example_tfrecord_data_{number}.png")
            plt.show()
            number += 1
            print("")


def main(download_folder=None, num_examples=None, top_k=5000, test_size=0.2, save_TFRecord_dir="image_text_TFRecord"):
    """
    :param download_folder: str, None->Automatic download of files
    :param num_examples: int, Select the first num_examples from the shuffled set, None for all data
    :param top_k:  int, Choose the top top_k words from the vocabulary
    :param test_size: float, test data number : train data number = test_size : 1
    :param save_TFRecord_dir: str, Store processed image and text TFRecordfile dir
    :return:
    """
    # Download files
    annotation_file, img_file_dir = dowload_MS_COCO_raw_data(download_folder=download_folder)

    # read_raw_image_and_text_file
    text_content_list, image_path_list = read_raw_image_and_text_file(annotation_file, img_file_dir,
                                                                      num_examples=num_examples)

    # Preprocess text
    text_content_list = preprocess_and_tokenize_texts(text_content_list, top_k)

    # split_and_save file
    store_image_path_text_content_TFRecord(image_path_list, text_content_list, test_size, save_TFRecord_dir)


if __name__ == "__main__":
    download_folder = "/home/b418a/disk1/pycharm_room/yuanxiao/my_lenovo_P50s/Image_captioning"
    # main(download_folder=download_folder, num_examples=None, top_k=5000,
    #      test_size=0.2, save_TFRecord_dir="image_text_TFRecord")

    check_tfrecord_data(image_TFRecord_path="./image_text_TFRecord/train_image_path.tfre",
                        text_TFRecord_path="./image_text_TFRecord/train_text_content.tfre",
                        text_tokenizer_path="text_tokenizer")

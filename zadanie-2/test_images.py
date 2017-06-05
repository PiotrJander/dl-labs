import tensorflow as tf

from train import augment_image, revert_transformations


def test_augmentation():
    # image = tf.constant([[[1, 2, 3], [4, 5, 6], [6, 7, 8]],
    #                      [[1, 2, 3], [4, 5, 6], [6, 7, 8]],
    #                      [[1, 2, 3], [4, 5, 6], [6, 7, 8]]])
    image = tf.constant([
        [
            [1, 2],
            [3, 4]
        ],
        [
            [5, 6],
            [7, 8]
        ]
    ])
    aug = augment_image(image)
    rev = revert_transformations(aug)
    minus = tf.subtract(image, tf.reduce_mean(rev, 0))
    with tf.Session() as sess:
        sess.run(tf.Print(minus, [minus], summarize=64))

# Typical setup to include TensorFlow.
import base64
import datetime
import tensorflow as tf
import os

LOG_DIR = 'logs/' + datetime.datetime.now().strftime("%B-%d-%Y;%H:%M")
# DATA_SET_SIZE = 10593
DATA_SET_SIZE = 20
IMAGES_DIR = '/data/spacenet2/images/'
HEATMAPS_DIR = '/data/spacenet2/heatmaps/'

# Make a queue of file names including all the JPEG images files in the relative
# image directory.
# images_filename_queue = tf.train.string_input_producer(
#     tf.train.match_filenames_once("/data/spacenet2/images/*.jpg"), shuffle=False)
# heatmaps_filename_queue = tf.train.string_input_producer(
#     tf.train.match_filenames_once("/data/spacenet2/heatmaps/*.jpg"), shuffle=False)

# Read an entire image file which is required since they're JPEGs, if the images
# are too large they could be split in advance to smaller files or use the Fixed
# reader to split up the file.
# image_reader = tf.WholeFileReader()


def read_and_decode(folder, f):
    with open(os.path.join(folder, f)) as file:
        # string = base64.b64encode(file.read())
        return tf.image.decode_jpeg(file.read(), ratio=2)


images = []
heatmaps = []

for _, imagename, heatmapname in zip(xrange(DATA_SET_SIZE), sorted(os.listdir(IMAGES_DIR)), sorted(os.listdir(HEATMAPS_DIR))):
    print imagename
    print heatmapname
    images.append(read_and_decode(IMAGES_DIR, imagename))
    heatmaps.append(read_and_decode(HEATMAPS_DIR, heatmapname))

images = tf.stack(images)
heatmaps = tf.stack(heatmaps)


# Read a whole file from the queue, the first returned value in the tuple is the
# filename which we are ignoring.
# images = []
# heatmaps = []
# for i in xrange(DATA_SET_SIZE):
#     images.append(read_and_decode(images_filename_queue))
#     heatmaps.append(read_and_decode(heatmaps_filename_queue))
# images = tf.stack(images)
# heatmaps = tf.stack(heatmaps)

images_summary_op = tf.summary.image("images", images[:10])
heatmaps_summary_op = tf.summary.image("heatmaps", heatmaps[:10])
all_summaries = tf.summary.merge_all()

# Decode the image as a JPEG file, this will turn it into a Tensor which we can
# then use in training.
# image = tf.image.decode_jpeg(image_file)
# image_batch = tf.expand_dims(image, 0)
# summary_op = tf.summary.image("plot", image_batch)
# all_summaries = tf.summary.merge_all()

# Start a new session to show example output.
with tf.Session() as sess:
    writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

    # Required to get the filename matching to run.
    tf.initialize_all_variables().run()

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Get an image tensor and print its value.
    # image_tensor = sess.run([image])
    # print(image_tensor)

    # shape = sess.run(tf.shape(images))
    # print shape

    summary = sess.run(all_summaries)
    writer.add_summary(summary)
    writer.close()

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)

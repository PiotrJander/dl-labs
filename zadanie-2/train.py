# Typical setup to include TensorFlow.
import datetime
import tensorflow as tf

LOG_DIR = 'logs/' + datetime.datetime.now().strftime("%B-%d-%Y;%H:%M")
DATA_SET_SIZE = 10593

# Make a queue of file names including all the JPEG images files in the relative
# image directory.
images_filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("/data/spacenet2/images/*.jpg"), shuffle=False)
heatmaps_filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("/data/spacenet2/heatmaps/*.jpg"), shuffle=False)

# Read an entire image file which is required since they're JPEGs, if the images
# are too large they could be split in advance to smaller files or use the Fixed
# reader to split up the file.
image_reader = tf.WholeFileReader()


def read_and_decode(queue):
    file = image_reader.read(queue)[1]
    return tf.image.decode_jpeg(file, ratio=0.5)


# Read a whole file from the queue, the first returned value in the tuple is the
# filename which we are ignoring.
images = []
heatmaps = []
for i in xrange(DATA_SET_SIZE):
    images.append(read_and_decode(images_filename_queue))
    heatmaps.append(read_and_decode(heatmaps_filename_queue))

# Decode the image as a JPEG file, this will turn it into a Tensor which we can
# then use in training.
# image = tf.image.decode_jpeg(image_file)
# image_batch = tf.expand_dims(image, 0)
# summary_op = tf.summary.image("plot", image_batch)
# all_summaries = tf.summary.merge_all()

# Start a new session to show example output.
with tf.Session() as sess:
    # writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

    # Required to get the filename matching to run.
    tf.initialize_all_variables().run()

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    s1, s2 = sess.run([tf.shape(images), tf.shape(heatmaps)])
    print s1, s2

    # Get an image tensor and print its value.
    # image_tensor = sess.run([image])
    # print(image_tensor)

    # summary = sess.run(all_summaries)
    # writer.add_summary(summary)
    # writer.close()

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)

need to read images to memory

now tf can convert a batch of images from jpg to tensors
and scale down at the same time

we could keep those as a tf variable

subsequently, we could have a batch train op
which takes a batch and does sgd on it

or we can do it in numpy and then feed dict
we could have an op for getting a batch from
the tf variable and then feeding it to another
train op

ok so it is valid to keep images as tf variable

but we need to init that var

we could pass a tensor (list) of strings

ok I'm stuck on this one

I need a ready solution

ok we're encouraged to load all images to memory

I just need to figure out how to

idea: shuffle while reading

use tf split to partition

how about this: we don't shuffle just yet

TODO read just one image and display in tf board; or two
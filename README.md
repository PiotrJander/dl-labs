now we'd like to take batches

make batch an op

take batch from train and validate

write images to summary

here the object is a singleton
used as a namespace

why not simply use global namespace of file?

or of function

now batch relies on queues

but queues are not needed maybe

what if we take 

have a counter var

take slice of 50 images in each iter

then do c += 50

one epoch = a number of batches

could also pass the counter as placeholder

but write summary first








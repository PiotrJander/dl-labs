TODO save model
TODO rescale to original resolution of 650
TODO visualize results

TODO using titan from laptop: prevent breaking connection
TODO functions like augment_image: back and forth between
    python and tf backend
TODO how exactly is memory used? as we resize, do we
    duplicate mem? possible to see mem consumption 
    in slurm or from inside Python?
TODO slurm: tell apart short and normal queue?
TODO opt batch size
TODO fewer channels - only three classes
TODO why tensorboard lost info about tensor shapes in up components?
TODO should we use queues
TODO should we randomize batches
TODO name scope vs var scope; need all explicit var scopes?
TODO For conv transpose use kernel size divisible by stride, or (possibly better option) upsample by nearest neighbour and then do regular convolution
TODO bias in conv nets - with/without batch norm
TODO how to reuse variables?
TODO use aws / gcloud, how to train on GPUs using virtual machines
TODO norm images to have zero mean and unit var
TODO take 256 crops (patches) of images

TODO end after 4 hrs or after validation accuracy 0.15
TODO rescale to original resolution of 650
TODO visualize results
TODO use two images in batch, maybe more

very simple:
* visualize what happened in the 20 examples case; summary of validation images
* concat image, ground truth, and prediction
* if can't log many images, combine into one
* even simpler: print just one image



TODO observe loss function plot, maybe log
TODO maybe one augmentation
TODO batch norm and m vs m - 1, or m / m - 1
TODO do we need regularization? plot distributions in different places
    and write sums of weighs
TODO what is the correct loss at chance performance?
TODO try adam with 10e-5
TODO plot loss / epochs or loss / batches
TODO check that validation accuracy almost as good as train acc
TODO for each layer, ration updates to weights, might be 1e-3
TODO histograms of activation / gradient per layer
TODO visualize first layer
TODO test adam with different hyperparams, compare with pure SGD
TODO need regularization when using batch norm?
TODO learning rate decay (Adam hyperparam) schedule
TODO test a log range of hyperparams, careful for border
TODO random search for hyperparams
TODO coarse to fine
TODO monitor especially first layer weights
TODO Decay your learning rate over the period of the training. For example, halve the learning rate after a fixed number
 of epochs, or whenever the validation accuracy tops off.
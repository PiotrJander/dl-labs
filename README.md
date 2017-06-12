






















TODO rescale to original resolution of 650
TODO visualize results
TODO use two images in batch, maybe more
TODO add augmentations

Q combine batch norm with dropout or L2 loss?
    batch norm -> initialization
    dropout, L2 -> regularization


first
TODO histogram for activations
TODO use w = np.random.randn(n) * sqrt(2.0/n), where n x no inputs to unit (neuron)



then
TODO check that validation accuracy almost as good as train acc
TODO histograms of activation / gradient per layer
TODO need regularization when using batch norm?



maybe
TODO for each layer, ratio of updates to weights, might be 1e-3
TODO batch norm and m vs m - 1, or m / m - 1
TODO test adam with different hyperparams, compare with pure SGD
TODO learning rate decay (Adam hyperparam) schedule
TODO test a log range of hyperparams, careful for border
TODO random search for hyperparams
TODO coarse to fine
TODO Decay your learning rate over the period of the training. For example, halve the learning rate after a fixed number
 of epochs, or whenever the validation accuracy tops off.
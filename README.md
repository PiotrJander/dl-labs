5. should we modify for test time?
6. batch norm in conv layer
7. make number of layers easily modifiable
8. compare speed of learning and accuracy
    on plots
9. maybe put BN after relu!


now add batch norm as in the example
now on to conv layers


conv layer:
filter / kernel
feature (map)
volume / kernel

in batch of size n, n x (a x b x c) volumes

idea: in moments: the axes specifies
in what direction to the the mean!


maybe test what we have
-> interesting, adding batch norm to one layer
    causes smaller accuracy
    
    
batch norm between conv and relu

suppose image is 28 x 28

and 32 filters / features, so 28 vs 28 vs 32

this is how many activations there are on EACH example

and there are 100 examples in batch

but examples share weights

for each of 32 filters, there are

in the filter, there are 5x5x32x64 weights

and these weights are shared between all the locations

take the simple case: 5x5x1x32 weights

in ffl, we were taking mean and variance for each dimension / feature

eg 1024 outputs to activation

this many means


parameters (γ and β) are learnt 
per feature map instead of per activation

feature map: 64 features! eureka

we have a volume x batch

would like to normalize along all axes other than depth
(depth = number of features)

in a cube, aggregating over layers
gives depth times mean, var

we could flatten the tensor, but we want to keep
the shape

think of the cube for now

































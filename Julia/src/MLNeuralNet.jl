using Mocha
include("utility.jl")
function MLLearnNeuralNet()
    (x_train,y_train,x_test,y_test,testingSetPos,testingSetNeg) = GatherTestCase()
    data  =  MemoryDataLayer(name="train-data",tops=[:data,:label],data=Array[x_train,y_train],batch_size=50)
    #conv  = ConvolutionLayer(name="conv1",n_filter=20,kernel=(5,5),bottoms=[:data],tops=[:conv])
    #pool  = PoolingLayer(name="pool1",kernel=(2,2),stride=(2,2),bottoms=[:conv],tops=[:pool])
    #conv2 = ConvolutionLayer(name="conv2",n_filter=50,kernel=(5,5),bottoms=[:pool],tops=[:conv2])
    #pool2 = PoolingLayer(name="pool2",kernel=(2,2),stride=(2,2),bottoms=[:conv2],tops=[:pool2])
    fc1   = InnerProductLayer(name="ip1",output_dim=5000,neuron=Neurons.LReLU(),bottoms=[:data], tops=[:ip1])#,weight_init=GaussianInitializer(std=10))
    #dr1   = DropoutLayer(name="dr1", ratio=0.5, bottoms=[:ip1]) #,
    fc2   = InnerProductLayer(name="ip2",output_dim=2000,neuron=Neurons.LReLU(),bottoms=[:ip1], tops=[:ip2])
    fc3   = InnerProductLayer(name="ip3",output_dim=500,neuron=Neurons.LReLU(),bottoms=[:ip2],tops=[:ip3])
    fc4   = InnerProductLayer(name="ip4",output_dim=100,neuron=Neurons.LReLU(),bottoms=[:ip3],tops=[:ip4])
    fc5   = InnerProductLayer(name="ip5",output_dim=30,neuron=Neurons.LReLU(),bottoms=[:ip4],tops=[:ip5])
    fc6   = InnerProductLayer(name="ip6",output_dim=2,neuron=Neurons.LReLU(),bottoms=[:ip5],tops=[:final])
    
    loss  = SoftmaxLossLayer(name="loss",bottoms=[:final,:label])
    backend = DefaultBackend()
    init(backend)
    common_layers = [fc1,fc2,fc3,fc4,fc5,fc6]
    net = Net("TweetDetect-train", backend, [data, common_layers...,loss])
    
    exp_dir = "snapshots"
    solver_method = SGD()
    params = make_solver_parameters(solver_method, max_iter=6000, regu_coef=0.0005,
    mom_policy=MomPolicy.Fixed(0.8),
   # lr_policy=LRPolicy.Fixed(0.1))
    #lr_policy=LRPolicy.Step(0.007, 0.1,250))
    lr_policy=LRPolicy.Inv(0.01, 0.0001, 0.75))
    solver = Solver(solver_method, params)
    
    #setup_coffee_lounge(solver, save_into="$exp_dir/statistics.jld", every_n_iter=1000)
    
    #report training progress every 100 iterations
    add_coffee_break(solver, TrainingSummary(), every_n_iter=500)
    
    # save snapshots every 5000 iterations
    #add_coffee_break(solver, Snapshot(exp_dir), every_n_iter=5000)
    
    # show performance on test data every 1000 iterations
    #data_test = HDF5DataLayer(name="test-data",source="test-data-list.txt",batch_size=100)
    
    data_test =  MemoryDataLayer(name="train-data",data=Array[x_train, y_train],batch_size=50)
    accuracy = AccuracyLayer(name="train-accuracy",bottoms=[:final, :label])
    #netplot = BinaryNetPlotLayer(name="train-net",bottoms=[:final, :label])
    #rocplot = ROCPlotLayer(name="train-net",bottoms=[:final, :label])
    test_train_net = Net("TweetDetect-train", backend, [data_test, common_layers..., accuracy])
    add_coffee_break(solver, ValidationPerformance(test_train_net), every_n_iter=1000)
    
    data_test =  MemoryDataLayer(name="test-data",data=Array[x_test, y_test],batch_size=50)
    accuracy = ConfusionMatrixLayer(name="test-accuracy",bottoms=[:final, :label])
    outputLayer = MemoryOutputLayer(name="test-Output",bottoms=[:final])
    netplot = BinaryNetPlotLayer(name="test-net",bottoms=[:final, :label])
    #rocplot = ROCPlotLayer(name="test-net",bottoms=[:final, :label])
    test_net = Net("TweetDetect-test", backend, [data_test, common_layers..., accuracy,netplot,outputLayer])
    add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=6000)
    
    data_test =  MemoryDataLayer(name="testPos-data",data=Array[x_test[:,testingSetPos], y_test[testingSetPos]],batch_size=50)
    confMatrix = ConfusionMatrixLayer(name="testPos-accuracy",bottoms=[:final, :label])
    accuracy = AccuracyLayer(name="testPos-accuracy",bottoms=[:final, :label])
#    netplot = BinaryNetPlotLayer(name="testPos-BinaryNetPlot",bottoms=[:final, :label])
#    outputLayer = MemoryOutputLayer(name="testPos-Output",bottoms=[:final])
    test_pos_net = Net("TweetDetect-testPos", backend, [data_test, common_layers..., accuracy,confMatrix]) 
    add_coffee_break(solver, ValidationPerformance(test_pos_net), every_n_iter=2000)
    
    data_test =  MemoryDataLayer(name="testNeg-data",data=Array[x_test[:,testingSetNeg], y_test[testingSetNeg]],batch_size=50)
    accuracy = AccuracyLayer(name="testNeg-accuracy",bottoms=[:final, :label])
    test_neg_net = Net("TweetDetect-testNeg", backend, [data_test, common_layers..., accuracy])
    add_coffee_break(solver, ValidationPerformance(test_neg_net), every_n_iter=2000)
    
    #Grabbing the output and analyze based on hashtag.

    solve(solver, net) 
    net_output = solver.coffee_lounge.coffee_breaks[3].coffee.validation_net.states[end].outputs

    destroy(net)
    destroy(test_train_net)
    destroy(test_net)
    destroy(test_pos_net)
    destroy(test_neg_net)
    shutdown(backend)
    
    test_size = length(y_test)
    startidx = test_size+50 - (test_size %50)+ 1
    net_output = reduce(hcat,net_output_in[end])[:,startidx:end]
    net_output = (findmax(net_output,1)[2] % 3).== 2
    net_output = vcat(net_output',zeros(8))
    
    return (net_output, y_test)
end

(net_output_in, y_test) = MLLearnNeuralNet()
TestTest(net_output_in, y_test,"NeuralNet");

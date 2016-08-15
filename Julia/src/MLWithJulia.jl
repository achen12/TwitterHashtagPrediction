#Do ML by picking up data from automated conversion.
#Utilizes achen12/Mocha.jl fork.
using JLD
using Mocha

function  setupVectorMultiClass(x,y)
    #Find balance Sets of labels
    histo_y_train = hist(y_train,32)
    minSetCount = findmin(histo_y_train[2])[1]
    #maxSetCount = findmax(histo_y_train[2])[1]
    idy = [ find(x -> x == i,y_train)[1:minSetCount]  for i in  collect(histo_y_train[1])[2:end]]
    return reduce(vcat,idy)
end


function  setupVectorBinary(x,y)
    #Find balance Sets of labels
    idySetPos = vcat(find(i-> i> 0,y))
    idySetNeg = rand(vcat(find(i-> i==0,y)),length(idySetPos))
    idy = vcat(idySetPos, idySetNeg) 
    shuffle!(idy)
     
    #Select Features
    x = x[1:6:96,idy]
    y = y[idy]
    y = y.> 0
    #Normalize X-Vector
    x = x ./ sum(x,1);
   
    return (Array{Float32,2}(x),Array{Float32,1}(y))
end


function  normalizeVector(x_vectors)

end
function MLLearn()
    vcatTuple = (i,j) -> (hcat(i[1],j[1]),vcat(i[2],j[2]))
    trainingFileSet = map(i -> "16-07-0" * string(i) * ".jld" , collect(2:8))
    (x_train, y_train) = reduce(vcatTuple, map(i -> setupVectorBinary(load(i,"x"),load(i,"y"))   
                                                 ,trainingFileSet))
    testingFile = "16-07-09.jld"
    (x_test,y_test) = setupVectorBinary(load(testingFile,"x"),load(testingFile,"y"))

    testingSetPos = vcat(find(y-> y>0,y_test))
    testingSetNeg = vcat(find(y-> y==0,y_test))
    
    
    data  =  MemoryDataLayer(name="train-data",tops=[:data,:label],data=Array[x_train,y_train],batch_size=1000)
    #conv  = ConvolutionLayer(name="conv1",n_filter=20,kernel=(5,5),bottoms=[:data],tops=[:conv])
    #pool  = PoolingLayer(name="pool1",kernel=(2,2),stride=(2,2),bottoms=[:conv],tops=[:pool])
    #conv2 = ConvolutionLayer(name="conv2",n_filter=50,kernel=(5,5),bottoms=[:pool],tops=[:conv2])
    #pool2 = PoolingLayer(name="pool2",kernel=(2,2),stride=(2,2),bottoms=[:conv2],tops=[:pool2])
    fc1   = InnerProductLayer(name="ip1",output_dim=1000,neuron=Neurons.LReLU(),bottoms=[:data],
                              tops=[:ip1])
    fc2   = InnerProductLayer(name="ip2",output_dim=100,neuron=Neurons.LReLU(),bottoms=[:ip1],
                              tops=[:ip2])
    fc3   = InnerProductLayer(name="ip3",output_dim=500,neuron=Neurons.LReLU(),bottoms=[:ip2],tops=[:ip3])

    fc4   = InnerProductLayer(name="ip4",output_dim=200,neuron=Neurons.LReLU(),bottoms=[:ip3],tops=[:ip4])
    
    fc5   = InnerProductLayer(name="ip5",output_dim=100,neuron=Neurons.ReLU(),bottoms=[:ip4],tops=[:ip5])
    fc6   = InnerProductLayer(name="ip6",output_dim=3,neuron=Neurons.ReLU(),bottoms=[:ip5],tops=[:final])
    
    loss  = SoftmaxLossLayer(name="loss",bottoms=[:final,:label])
    
    backend = DefaultBackend()
    init(backend)
    
    common_layers = [fc1,fc2,fc3,fc4,fc5,fc6]
    net = Net("TweetDetect-train", backend, [data, common_layers..., loss])
    
    exp_dir = "snapshots"
    solver_method = SGD()
    params = make_solver_parameters(solver_method, max_iter=3000, regu_coef=0.0005,
        mom_policy=MomPolicy.Fixed(0.90),
   ;# lr_policy=LRPolicy.Fixed(0.1))
        lr_policy=LRPolicy.Inv(0.01, 0.001,0.75))
    solver = Solver(solver_method, params)
    
    #setup_coffee_lounge(solver, save_into="$exp_dir/statistics.jld", every_n_iter=1000)
    
    #report training progress every 100 iterations
    add_coffee_break(solver, TrainingSummary(), every_n_iter=100)
    
    # save snapshots every 5000 iterations
    #add_coffee_break(solver, Snapshot(exp_dir), every_n_iter=5000)
    
    # show performance on test data every 1000 iterations
    #data_test = HDF5DataLayer(name="test-data",source="test-data-list.txt",batch_size=100)
    
    data_test =  MemoryDataLayer(name="train-data",data=Array[x_train[:,1:10000], y_train[1:10000]],batch_size=50)
    accuracy = ConfusionMatrixLayer(name="train-accuracy",bottoms=[:final, :label])
    netplot = BinaryNetPlotLayer(name="train-net",bottoms=[:final, :label])
    test_net = Net("TweetDetect-test", backend, [data_test, common_layers..., accuracy,netplot])
    add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=500)
    
    
    #data_test =  MemoryDataLayer(name="test-data",data=Array[x_test[20000:40000], y_test[20000:40000]],batch_size=50)
    #accuracy = AccuracyLayer(name="test-accuracy",bottoms=[:ip2, :label])
    #test_net = Net("TweetDetect-test", backend, [data_test, common_layers..., accuracy])
    #add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=100)
    
    data_test =  MemoryDataLayer(name="testPos-data",data=Array[ x_test[:,testingSetPos], y_test[testingSetPos]],batch_size=50)
#    confMatrix = ConfusionMatrixLayer(name="testPos-accuracy",bottoms=[:final, :label])
    accuracy = AccuracyLayer(name="testPos-accuracy",bottoms=[:final, :label])
#    netplot = BinaryNetPlotLayer(name="testPos-BinaryNetPlot",bottoms=[:final, :label])
    test_pos_net = Net("TweetDetect-testPos", backend, [data_test, common_layers..., accuracy])
    add_coffee_break(solver, ValidationPerformance(test_pos_net), every_n_iter=3000)
    

    data_test =  MemoryDataLayer(name="testNeg-data",data=Array[x_test[:,testingSetNeg], y_test[testingSetNeg]],batch_size=50)
    accuracy = AccuracyLayer(name="testNeg-accuracy",bottoms=[:final, :label])
    test_neg_net = Net("TweetDetect-testNeg", backend, [data_test, common_layers..., accuracy])
    add_coffee_break(solver, ValidationPerformance(test_neg_net), every_n_iter=500)
    
    solve(solver, net) 
    destroy(net)
    destroy(test_net)
    destroy(test_pos_net)
    destroy(test_neg_net)
    shutdown(backend)
end

MLLearn()

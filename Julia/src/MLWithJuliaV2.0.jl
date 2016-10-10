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

function normalizeByTrendNoTrend(x)
    #Baseline Normalization
    x = x ./ sum(x,1)

    #spikeNormalization
    x = abs(x - vcat(zeros(1,size(x,2)),x[1:(size(x,1)-1),:]))
    alpha = 1.2
    x = x .^  alpha

    #Smoothing
    x = cumsum(x,1)
    return x
end

function  setupTrainingVectorBinary(x,y)
    #Find balance Sets of labels
    #idySetPos = vcat(find(i-> (i== 30) || (i == 29),y))
    idySetPos = vcat(find(i-> (i.>15),y))
    idySetNeg = rand(vcat(find(i-> (i .==0),y)),length(idySetPos))
    idy = vcat(idySetPos, idySetNeg) 
    shuffle!(idy)
    #Select Features
    x1 = x[1:7:112,idy] 
    x1 = normalizeByTrendNoTrend(x1)
    x2 = x[2:7:112,idy] ./ 2^14
    x = vcat(x1,x2)
    #x = log(log(x+10))
    y = y[idy]
    y = y .> 15
    #Normalize X-Vector
    println(size(x))
    #x = vcat(x ./ sum(x,1),log(findmax(x,1)[1])) #.- 0.5;
    #x = x ./ findmax(x,1)[1] #.- 0.5;
    return (Array{Float64,2}(full(x)),Array{Float64,1}(full(y)[:,1]))
end

function  setupVectorBinary(x,y)
    #Select Features
    x1 = x[1:7:112,:] 
    x1 = normalizeByTrendNoTrend(x1)
    x2 = x[2:7:112,:] ./ 2^14
    x = vcat(x1,x2)
    #x = log(log(x+10))
    y = y .>  15
    #y = (y .==  1) + (y .== 2 )
    #Normalize X-Vector
    #x = vcat(x ./ sum(x,1),log(findmax(x,1)[1])) #.- 0.5;
    return (Array{Float64,2}(full(x)),Array{Float64,1}(full(y)[:,1]))
end
# SVM k-NN XGBoost Tree-based
function MLLearn()
    vcatTuple = (i,j) -> (hcat(i[1],j[1]),vcat(i[2],j[2]))
    trainingFileSet = map(i -> "Step1_Parse/MLVectors/16-08-" * dec(i,2) * ".jld" , collect(1:20))
    (x_train, y_train) = reduce(vcatTuple, map(i -> setupTrainingVectorBinary(SparseMatrixCSC{UInt64,Int64}(load(i,"x")),load(i,"y")),trainingFileSet))
    testingFile = "Step1_Parse/MLVectors/16-07-02.jld"
    (x_test,y_test) = setupVectorBinary(SparseMatrixCSC{UInt64,Int64}(load(testingFile,"x")),load(testingFile,"y"))

    testingSetPos = vcat(find(y-> y>0,y_test))
    testingSetNeg = vcat(find(y-> y==0,y_test))[1:10000]
    println("TestingTest dimension", length(testingSetPos),length(testingSetNeg)) 
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
    return (net_output, y_test)
end
function TestTest(net_output_in,y_test)
    test_size = length(y_test)
    startidx = test_size+50 - (test_size %50)+ 1
    net_output = reduce(hcat,net_output_in[end])[:,startidx:end]
    net_output = (findmax(net_output,1)[2] % 3).== 2
    net_output = vcat(net_output',zeros(8))
    println(sum(findnz(net_output)[3]))
    println(sum(y_test))

    #Grabbing Truths (y)
    z_test = load("Step1_Parse/MLVectors/16-07-02.jld","z")
    totalSum = size(z_test)[1] #Total count of hashtags
    y_test = load("Step1_Parse/MLVectors/16-07-02.jld","y")
    y_test = find( full(y_test)[:,1] .== 30)
    z3_test = map(i -> !isempty(searchsorted(y_test,i)), findnz(z_test)[3])
    z_test = findnz(z_test)
    z3_test = sparse(z_test[1],z_test[2],z3_test)
    truth = unique(findnz(z3_test)[1])
    
    #Create Truth Timing
    #TODO
    temp = findn(z3_test)
    truthTiming = Dict(zip(temp[1],temp[2]))

    #Grabbing Predicted Truths (hat{y})
    #Assumign net_ouput to be Array{Boolean}
    net_output = find(net_output)
    hz3_test = map(i -> !isempty(searchsorted(net_output,i)), z_test[3])
    hz = sparse(z_test[1],z_test[2],hz3_test)
    predicted_truth = Array{Number,1}()
    declarationTiming = Dict{Number,Int}()
    repetit_size = 5
    #Sliding Window
    for i in (repetit_size+1):size(hz,2)
        foundTruth = find(sum(hz[:,(i-repetit_size):i],2) .>=(repetit_size  ))
        append!(predicted_truth,foundTruth)
        for x in foundTruth
            if ! (x in keys(declarationTiming))
                declarationTiming[x] = i
            end
        end
    end
    predicted_truth  =  unique(predicted_truth)
    #Generating statistics for TruePos, FalsePos, FalseNeg, and TrueNeg.
    sort!(truth)
    sort!(predicted_truth)
    println(length(truth))
    println(length(predicted_truth))
    truePos = intersect(truth,predicted_truth)
    falsePos = setdiff(predicted_truth,truth)
    falseNeg = setdiff(truth,predicted_truth)
    truePos = length(truePos)
    falsePos = length(falsePos)
    falseNeg = length(falseNeg)
    trueNeg = totalSum - truePos - falsePos - falseNeg 
    print("True Positives Count:")
    println(truePos)
    print("True Negative Count:")
    println(trueNeg)
    print("False Positives Count:")
    println(falsePos)
    print("False Negative Count:")
    println(falseNeg)

    println("Statistics VS TnT")
    println("TruePositiveRate: ",truePos./(truePos+falseNeg), " vs TnT 95%")
    println("FalsePositiveRate: ",falsePos./(falsePos+trueNeg), " vs TnT 4%")
    timingStatistics = Array{Int,1}()
    for x in keys(truthTiming)
        if x in keys(declarationTiming)
            # Do also include the delta of the first sight compare with the true Trend Declaration in Twitter 
            push!(timingStatistics,declarationTiming[x]-(truthTiming[x]))
        end
    end
    #Create Histogram
    PyPlot.clf()
    yy = hist(timingStatistics,40)
    bar(yy[1][1:end-1]./2,yy[2],10)
    title("Trend Prediction Timing of True Postive hashtags of 07-02-2016")
    ylabel("Count of Hashtags")
    xlabel("Predicted Time - Truth Time (Minutes)")
    savefig("histo.png")
    PyPlot.clf()



    return timingStatistics 
end
#(net_output, y_test) = MLLearn()

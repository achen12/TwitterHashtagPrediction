using JLD
using PyPlot

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
    x1 = x[1:6:96,idy] 
    x1 = normalizeByTrendNoTrend(x1)
    #x2 = x[2:6:96,idy] ./ 2^14
    x = x1#vcat(x1,x2)
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
    x1 = x[1:6:96,:] 
    x1 = normalizeByTrendNoTrend(x1)
    #x2 = x[2:6:96,:] ./ 2^14
    x = x1 #vcat(x1,x2)
    #x = log(log(x+10))
    y = y .>  15
    #y = (y .==  1) + (y .== 2 )
    #Normalize X-Vector
    #x = vcat(x ./ sum(x,1),log(findmax(x,1)[1])) #.- 0.5;
    return (Array{Float64,2}(full(x)),Array{Float64,1}(full(y)[:,1]))
end

#Test,Train range definition
trainingFileSet = map(i -> "../../../../16-07-" * dec(i,2) * ".jld" , collect(2:3))
testingFile = "../../../../16-07-01.jld"

function GatherTestCase()
    vcatTuple = (i,j) -> (hcat(i[1],j[1]),vcat(i[2],j[2]))
    (x_train, y_train) = reduce(vcatTuple, map(i -> setupTrainingVectorBinary(SparseMatrixCSC{UInt64,Int64}(load(i,"x")),load(i,"y")),trainingFileSet))
    (x_test,y_test) = setupVectorBinary(SparseMatrixCSC{UInt64,Int64}(load(testingFile,"x")),load(testingFile,"y"))

    testingSetPos = vcat(find(y-> y>0,y_test))
    testingSetNeg = vcat(find(y-> y==0,y_test))[1:10000]
    println("TestingTest dimension", length(testingSetPos),length(testingSetNeg)) 
    return (x_train,y_train,x_test,y_test,testingSetPos,testingSetNeg)
end


function TestTest(net_output,y_test,suffix)
    
    println(sum(findnz(net_output)[3]))
    println(sum(y_test))

    #Grabbing Truths (y)
    z_test = load(testingFile,"z")
    totalSum = size(z_test)[1] #Total count of hashtags
    y_test = load(testingFile,"y")
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
    bar(yy[1][1:end-1].*2,yy[2],10) #Every slot is 2 minutes
    title("Trend Prediction Timing of True Postive hashtags of 07-01-2016")
    ylabel("Count of Hashtags")
    xlabel("Predicted Time - Truth Time (Minutes)")
    savefig("histo"*suffix*".png")
    PyPlot.clf()



    return timingStatistics 
end


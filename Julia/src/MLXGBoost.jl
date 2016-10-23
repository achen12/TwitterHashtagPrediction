#Using XGBoost
using XGBoost
include("utility.jl")
function MLLearnXGBoost()
    (x_train,y_train,x_test,y_test,testingSetPos,testingSetNeg) = GatherTestCase()

    num_round = 20
    #return x_train,y_train
    bst = xgboost(x_train', num_round, label=y_train, eta=0.9, max_depth=10)
    pred = XGBoost.predict(bst, x_test')
    #return pred
    println("test-error=", sum(Array{Int,1}(pred .> 0.5) .!= (y_test.>0.5)) / float(size(pred)[1]))
    
    return (sparse(Array{Int,1}(pred.>0.5)),y_test)
end
(net_output_in, y_test) =  MLLearnXGBoost()
TestTest(net_output_in, y_test,"XGBoost")

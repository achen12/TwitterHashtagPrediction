#Regression Based.
include("utility.jl")
using Regression
function MLLearnRegression()
    (x_train,y_train,x_test,y_test,testingSetPos,testingSetNeg) = GatherTestCase()
    
    n = length(x_train)   # number of samples

    # perform estimation
    ret = Regression.solve(
        logisticreg(x_train, sign(y_train-0.5); bias=1.0),   # construct a logistic regression problem
        reg=SqrL2Reg(1.0e-2),          # apply squared L2 regularization
    options=Regression.Options(verbosity=:iter, grtol=1.0e-6 * n))  # set options
    
    # extract results
    #w_e = ret.sol
    #net_output = w_e*x_test
    #net_output = solver.coffee_lounge.coffee_breaks[3].coffee.validation_net.states[end].outputs
    w_e = ret.sol
    net_output = (w_e[1:end-1]'*x_test .+ w_e[end]).>0
    return (net_output, y_test)
end
(net_output_in, y_test) = MLLearnRegression()
TestTest(net_output_in, y_test,"Regression");

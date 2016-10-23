using NearestNeighbors
include("utility.jl")
function MLLearnNN()
    
    (x_train,y_train,x_test,y_test,testingSetPos,testingSetNeg) = GatherTestCase()

    
    kdtree = KDTree(x_train)
    k = 5
    
    net_output = [(k*0.8).<sum(y_train[knn(kdtree,x_test[:,i],k,true)[1]]) for i in 1:length(y_test)]
    return (sparse(net_output),y_test)
    
end
(net_output_in,y_test) = MLLearnNN();
TestTest(net_output_in, y_test,"k-NN");

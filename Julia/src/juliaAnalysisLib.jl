using JSON
import GZip
#Spark-Like Functions
function mapByKey(f,d)
    nd = Dict{keytype(d),Any}()
    for k in keys(d)
        nd[k] = f(d[k])
    end
    return nd
end


function reduceByKeyAndWindow(f,win,slid,dstr)
    #Reduce by window and Key
    #Takes in a collection of values of a key and a certain window size (win:Size of window in array slots)
    #with slide factor (slid), and reduce by a function, (f:binary reduce function).
    #takes in key-value dstream (dstr: Array{Dict,1})
    
    #initialize new dstr
    newDstrSize = round(Int64,ceil(length(dstr)/slid))
    newDstr = Array{Dict,1}(newDstrSize)
    #for loop with sliding factor
    for i in collect(1:newDstrSize)
        #for each sliding loop creates an array of keys,
        currI = (i-1)*slid+1
        currKeys = reduce(vcat,map(x -> collect(keys(x)), dstr[max(currI-win,1):(currI)]))
        #subsequently populate an dict for each keys
        currDict = Dict{keytype(dstr[1]),valtype(dstr[1])}()
        for k in currKeys
            #collect all value in the window
            currKeyCol = Array{valtype(dstr[1]),1}()
            for j in collect(max(currI-win,1):currI)
                if haskey(dstr[j],k)
                    push!(currKeyCol,dstr[j][k])
                end
            end
            #Then reduce.
            currDict[k] = reduce(f,currKeyCol)
        end
        newDstr[i] = currDict
    end
    return newDstr
end

function mapReduceByKey(mapF,reduceF, dict)
    return Dict(map(x -> Pair(x[1],mapreduce(mapF,reduceF,x[2])),dict))
end

function recurrentFixedWindow(t,dstr)
    #construct numeric recurrent vector from a fixed length sliding window
    #with default values of zero
    #assume all vector within any single slot is of same length.
    #input is an array of Dicts of string to (numeric array or number)
    
    #First let's determine n the number of array of a single slot.
    n =  6#length(first(dstr[1])[2])
    #let's determine l, the number of windows the operation will result in.
    l = length(dstr) - t
    newdst = Array{Dict{keytype(dstr[1]),Array{Number,1}},1}(l)
    for i in collect(1:l)
        #TODO make windowed operation
        #create an array of possible keys
        currKeys = reduce(vcat,map(x -> collect(keys(x)), dstr[i:(i+t)]))
        currDict = Dict{keytype(dstr[1]),Array{Number,1}}()
        #for each of the keys we'll populate a recurrent vector.
        for k in currKeys
            currArr = Array{Number,1}()
            for j in collect(i:(i+t))
                if haskey(dstr[j],k)
                    currArr = vcat(currArr,dstr[j][k])
                else
                    #Note that default value is filled if the key doesn't exist within a slot.
                    append!(currArr,zeros(n))
                end
            end
            currDict[k]=currArr
        end
        newdst[i] = currDict
    end
    return newdst
end

function toDstream(inX,t) #t=120000
    println("Begin binning to 2 minutes slots")
    x= inX[findn(["created_at" in keys(y) for y in inX])];
    dt = join(split(x[1]["created_at"]," ")[[1,2,3,4,6]]," ")
    startTime = DateTime(dt,"e u d H:M:S y")
    dt = join(split(x[end]["created_at"]," ")[[1,2,3,4,6]]," ")
    endTime = DateTime(dt,"e u d H:M:S y")
    println(startTime)
    windowCount = 1+round(Int,ceil(Dates.value(endTime - startTime) /t ))#Divid into 2 minutes windows
    Tweets = Array{Array{Dict}}(windowCount)
    for i in collect(1:windowCount)
        Tweets[i] = Array{Dict,1}()
    end
    println(windowCount)
    for tweet in x
        dt = join(split(tweet["created_at"]," ")[[1,2,3,4,6]]," ")
        currTime = DateTime(dt,"e u d H:M:S y")
        windowIndex = round(Int,ceil(Dates.value(currTime - startTime) /t ))#Divid into 2 minutes windows 
        if ((windowIndex >= 0) && (windowIndex < (windowCount)))
            append!(Tweets[windowIndex+1],[tweet])
        end
    end
#    println(Tweets[1][1]) #First 10 sec Window's First tweet.
    return Tweets
end

function findHash(x)
    words = split(x," ")
    return words[find(word -> (length(word) > 1)&& (word[1] == '#'),words)]
end
#findHash(x[20]["text"])
#x[20]["text"]
function extractHashMap(x) #calculates per window
    z = mapreduce(i-> [(y,i) for y in findHash(i["text"])],vcat,x);
    # z is the table of hash to a list of tweets
    c = Dict{AbstractString,Array}();
    for i in z
        if !haskey(c,i[1])
            c[i[1]] = []
        end
        append!(c[i[1]],[i[2]])
    end
    return c
end
function toHashMap(Tweetdstr)
    println("Begin sorting into Hashes")
    HashTweetMap = map(extractHashMap,Tweetdstr)#Dict{Hash, Tweets} per 120 secs window
end
function toVectorDstr(HashTweetMap)
    println("Conversion to Vectors")
    vectorSize = 1
    vectorIndMapping = Array{Tuple{Int64,AbstractString},1}()
    x_vector = Array{Array{Number,1},1}()
    
    ##Volume
    function calculateVolume(x) #calculates per window for all hash [tweets] key pairs
        map(v -> length(v[2]),x)
    end
    HashVolumeMap = map(x-> mapByKey(length,x),HashTweetMap)
    #HashVolumeMap = reduceByKeyAndWindow((+),10,1,HashVolumeMap) #This is sum into overlapping 1200 secs windows
    
    ##Unique Users Count
    HashUserMap = map(z -> Dict(map(x-> Pair(x[1],unique(map(y-> y["user"]["name"],x[2]))),z)),HashTweetMap)
    #HashUserMap = reduceByKeyAndWindow(vcat,10,1,HashUserMap) #Organize into 1200 secs windows
    HashUserMap = map(z -> Dict(map(x-> Pair(x[1],length(unique(x[2]))),z)),HashUserMap)
    
    
    ## 1st degree Connected Tweets
    HashDegreeMap = map(z -> Dict(map(x-> Pair(x[1],unique(map(y-> findHash(y["text"]),x[2]))),z)),HashTweetMap)
    #HashDegreeMap = reduceByKeyAndWindow(vcat,10,1,HashDegreeMap) #Organize into 1200 secs windows
    HashDegreeMap = map(z -> Dict(map(x-> Pair(x[1],unique(x[2])),z)),HashDegreeMap)
    HashDegreeMap = map(z -> Dict(map(x-> Pair(x[1],mapreduce(y-> HashVolumeMap[z][y[1]],(+),vcat(x[2]))),HashDegreeMap[z])),1:length(HashDegreeMap))


    
    #Followers Sum Weighted by tweets
    
    function getFollower(i)
        if i["user"]["followers_count"] >0
            return i["user"]["followers_count"]
        else
            return 0
        end
    end

    HashFollowerMap = map(x-> mapByKey(y -> mapreduce(getFollower,(+),y) ,x),HashTweetMap)

    
    #Friends Sum Weighted by tweets

    function getFriends(i)
        if i["user"]["friends_count"] >0
            return i["user"]["friends_count"]
        else
            return 0
        end
    end


    HashFriendMap = map(x-> mapByKey(y -> mapreduce(getFriends,(+),y) ,x),HashTweetMap)



    #Retweeted volume
    function getRetweetVolume(i)
        if "retweeted_status" in keys(i)
            return i["retweeted_status"]["retweet_count"]
        else
            return 0
        end
    end
    HashRTVolumeMap = map(x-> mapByKey(y -> mapreduce(getRetweetVolume,(+),y) ,x),HashTweetMap)
       

 
    #Summarize into x_vector essentially flatmap into keyless
    HashVectorMap = Array{Dict{AbstractString,Any},1}(length(HashVolumeMap))
    
    for i in collect(1:length(HashVolumeMap))
        currDict = Dict{AbstractString,Any}()
        for k in keys(HashVolumeMap[i])
            currDict[k] = [HashUserMap[i][k],HashVolumeMap[i][k],HashDegreeMap[i][k],HashFollowerMap[i][k],HashFriendMap[i][k],HashRTVolumeMap[i][k]]
        end
        HashVectorMap[i] = currDict
    end
    
    return HashVectorMap
end
function toXVectors(VectorDStr,recurrentCount)
    x_vector = Array{Array{Number,1},1}()
    HashVectorMap = recurrentFixedWindow(recurrentCount,VectorDStr)
    newdict = Array{Dict{keytype(VectorDStr[1]),Integer},1}(length(HashVectorMap))
    for i in collect(1:length(HashVectorMap))
        newdict[i] = Dict{keytype(VectorDStr[1]),Integer}() 
    end
    for i in collect(1:length(HashVectorMap))
        for k in keys(HashVectorMap[i])
            push!(x_vector, HashVectorMap[i][k])
            newdict[i][k] = length(x_vector)
        end
    end
    return (x_vector, newdict)
end



function tohashFreq(HashTweetMap)
    return map(j->map(z -> Pair(z[1],map(i -> i["timestamp_ms"],z[2])),j),HashTweetMap)
end

#Hash Freq analysis
function freqAnalaysis(dateString)
    println("Begin analysis")
    trendx = mapreduce(x->map(y -> JSON.parse(y),readlines(GZip.open("Step0_Raw/Json/trend"*dateString*"/"*x))),vcat,readdir("Step0_Raw/Json/trend"*dateString));
    trendx = mapreduce(i -> mapreduce(j-> Pair(i["timestamp_ms"],j["Name"]),vcat,i["Trend"]),vcat,trendx)
    trendingHashes = unique(trendx[find(i -> i[2][1] == '#',trendx)]);
    trendingHashes = reduce(vcat, map(x-> hcat(x[2],x[1]),trendingHashes))
    println("found all trends")
    trendingHashtags = unique(trendingHashes[:,1]) #Important A list of trending hashes during this 24 hour window.
    trendingHashtagsFirstTimestamp = hcat(trendingHashes[map(i->findfirst(trendingHashes[:,1],i),trendingHashtags),:],collect(1:length(trendingHashtags)))
    writecsv("trendingHashtagsFirstTimestamp.csv",trendingHashtagsFirstTimestamp)
    println("Saved trending Hashes and their first time stamp")
    trendingHashtags = vcat(trendingHashtags,["#love";"#Pokemon";"#happy"]) #Adding non trending hashtags
    x_source = mapreduce(x->tohashFreq(toHashMap(toDstream(map(y -> JSON.parse(y),readlines(GZip.open("Step0_Raw/Json/"*dateString*"/"*x))),120000))),vcat,readdir("Step0_Raw/Json/"*dateString));
    println("Begin construct Freq Analysis for the chosen hashtags")
    z = map(k -> Pair(k,map(y-> y[2],mapreduce(x-> x[find(i -> i[1] == k,x)],vcat,x_source))),trendingHashtags);
    print(length(find(i -> ! isempty(i[2]),z))) 
    print("/")
    print(length(z))
    println(" Empty")
    z = z[find(i -> ! isempty(i[2]),z)];
    z = map(k -> Pair(k[1],reduce(vcat,k[2])),z);
    sol = mapreduce(z -> hcat(z[1],z[2]),vcat,mapreduce(x->map(i->(i,find(j -> j==x[1],trendingHashtags)[1]),x[2]),vcat ,z));
    writecsv("HashFreqAnalaysis.csv",sol)
    #Sol is set to chart hash freq 
end

using JLD
using JSON
import GZip

function convertToVectorFiles(dirString,dateString,windowSizeInMS = 120000,recurrentVectorSizeInWindowCount = 15,predictionCount = 30)
    f = jldopen("Step1_Parse/MLVectors/"*dateString*".jld","w")
    x_store = mapreduce(x->toVectorDstr(toHashMap(toDstream(map(y -> JSON.parse(y),readlines(GZip.open(dirString * "/" * dateString * "/" * x))),windowSizeInMS))),vcat,readdir(dirString * "/" * dateString))
    (x_vector,vectorIndMapping) = toXVectors(x_store,recurrentVectorSizeInWindowCount)
    x_store = 0
    trendx = mapreduce(x->map(y -> JSON.parse(y),readlines(GZip.open(dirString * "/trend" * dateString * "/" * x))),vcat,readdir(dirString * "/trend" * dateString)) #File Readin
    trendx = mapreduce(i -> mapreduce(j-> Pair(i["timestamp_ms"],j["Name"]),vcat,i["Trend"]),vcat,trendx)
    trendingHashes = unique(trendx[find(i -> i[2][1] == '#',trendx)])
    trendx = 0;
    recurrentCount = predictionCount
    #dt = Dates.datetime2unix(startTime)*1000
    dt = first(trendingHashes)[1]
    trendingHashes = map(x -> Pair(round(Int64, (x[1] - dt)/windowSizeInMS),x[2]), trendingHashes)
    y_vector = zeros(length(x_vector))
    println("Being labeling process")
    for x in trendingHashes
        #recurrentIndex = x[1] - recurrentCount
        #if (recurrentIndex > 0) #TODO check Sanity
            for i in collect(0:predictionCount)#Window to declare subsignal a trend from declaration.
                if (x[1]-i > 0) && (x[1]-i < length(vectorIndMapping)) && haskey(vectorIndMapping[x[1]-i],x[2])
                    if  y_vector[vectorIndMapping[x[1]-i][x[2]]] == 0 
	                y_vector[vectorIndMapping[x[1]-i][x[2]]] = 1+i
                    end
                end
            end
        #end
    end
    write(f,"x",sparse(Array{UInt32,2}(reduce(hcat,x_vector))))
    write(f,"y",Array{UInt8,1}(y_vector))
    print("Vector Count: ")
    println(length(x_vector))
    println("Prediction Label Summary:")
    println(hist(y_vector,predictionCount))
    convertToLIBSVMFile("Step1_Parse/MLVectors",dateString*".libsvm.data",x_vector,y_vector)
    x_vector =0
    y_vector =0
    #Create a new Hashtag Table:  List of hashtag "zh", and a sparse matrix of hashtag+ time to index "z".
    zh = unique(reduce(vcat,map( i-> collect(keys(i)),vectorIndMapping)))
    sort!(zh)
    z = spzeros(length(zh),length(vectorIndMapping))
    for i in collect(1:length(vectorIndMapping))
        for j in keys(vectorIndMapping[i])
            z[searchsortedfirst( zh,j),i] = vectorIndMapping[i][j]
        end 
    end
    println("Conversion success, Begin Saving")
    write(f,"zh", z)
    write(f,"z",z)
    close(f) 
end

##Save into LibSVM format
function convertToLIBSVMFile(dir::String, fileDir::String,x,y)
    open(dir *"/"*fileDir,"w") do f
        for i in collect(1:length(y))
            if(i%10000 ==0)
                print(".")
            end
            write(f,string(round(Int,y[i]))* " " *
            join(map(j -> string(j) * ":" * string(x[i][j]),find(x[i]))," ") *"\n")
        end
    end
end

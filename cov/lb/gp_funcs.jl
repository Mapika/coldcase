using Combinatorics
function DetermineOrbitNumbersQary(n)
    orbitNumbers= Dict();
    orbitNumber=0;
    ## instead of storing orbit representatives as triples [i,j,t,p] 
    ## in my code they are stored as quintuples [i1,i2,i3,i4,i5], where 
    ##
    # i1 = n-(i-t)-(j-t)-p-(t-p) is the number of positions x000
    # i2 = j-t                   is the number of positions x001  
    # i3 = i-t                   is the number of positions x010    
    # i4 = p;                    is the number of positions x011
    # i5 = t-p;                  is the number of positions x012
    
    function numberOrbit(i,j,t,p)
        for tprime=0:n, iprime=0:n, jprime=0:n, pprime=0:tprime  
            if n>=n-iprime-jprime+tprime>=0 && n>=jprime-tprime >=0 && n>= iprime-tprime >= 0  &&  (isequal(sort([i,j,i+j-t-p]), sort([iprime,jprime,iprime+jprime-tprime-pprime])) && t-p == tprime-pprime )
                orbitNumbers[[n-iprime-jprime+tprime, jprime-tprime, iprime-tprime, pprime, tprime-pprime]] =orbitNumber 
            end
        end
    end

    #first number the orbits x_{i,0}^{0,0} where i=0,...,n 
    for i=0:n 
        if !haskey(orbitNumbers, [n-i, 0, i, 0, 0])
            orbitNumber+=1
            numberOrbit(i,0,0,0)
        end
    end
    for t=0:n, i=0:n, j=0:n, p = 0 : t  
        if n>=n-i-j+t>=0 && n>=j-t >=0 && n>= i-t >= 0 && !haskey(orbitNumbers, [n-i-j+t, j-t, i-t, p, t-p])
            orbitNumber+=1
            numberOrbit(i,j,t,p)
        end
    end
    return orbitNumbers, orbitNumber
end

function DetermineBlockIndicesQ(n,k,a) 
    IndexSet=[]
    #rowindex: number of ones in first row of tableau 1
    for i=k:n+a-k
    #rowindex: number of ones in first row of tableau 2 
        push!(IndexSet,i);
    end
    return IndexSet, size(IndexSet,1)
end

function gammaprime(n, i2,i3,i4,i5,q=3)
    return big(q-1)^(i2+i3+i4+i5)*big(q-2)^(i5)*multinomial(big(i2),big(i3),big(i4),big(i5),big(n-i2-i3-i4-i5))
end

function beta(m,t,i,j,k)
    return sum([(-1)^(t-u)*binomial(big(u),t)*binomial(big(m-2*k),m-k-u)*binomial(big(m-k-u),i-u)*binomial(big(m-k-u),j-u) for u=0:m])
end

function alpha(n,i,j,t,p,a,k,q=3)
    start= beta(n-a,t-a,i-a,j-a,k-a)*big(q-1)^(i+j-t)
    start2=0;
    for g=0:p 
        if  t-a >=p-g && a >=0
            start2+=(-1)^(a-g)*binomial(big(a),g)*binomial(big(t-a),p-g)*big((q-2))^(t-a-p+g)
        else 
            start2+=0
        end
    end
    return start*start2
end
   
#Given words x,y with d(x,y)=(i,j,t,p), calculate for each j',t',p' the number
#Coef[j',t',p']:=sum_d lambda_d*
#|words z: d(y,z)=d, d(x,z)=(i,j',t',p')|.
#this is needed for inequality row x around word y.
##lambda is a vector of size n+1. lambda[dist+1] corresponds with lambda_dist from the paper. 
function MakeDistrQary(n,i,j,t,p,lambda,q=3)
	ti=i-t;
	tj=j-t;
    tp = t-p;
    rest=n+t-i-j
	coef=Dict();

    #add values
	for a1=0:ti, a2=0:ti-a1, b1=0:tj, b2=0:tj-b1, c1=0:p, c2=0:p-c1, d1=0:tp, d2=0:tp-d1, d3=0:tp-d1-d2, e=0:rest
        j2=a1+a2+b1+b2+c1+c2+d1+d2+d3+e;
        t2=a1+a2+c1+c2+d1+d2+d3;
        p2=a1+c1+d1
        dist=a1+a2+e+j-b1-c1-d2;	
        if haskey(coef,(j2,t2,p2))
            coef[j2,t2,p2]+=lambda[dist+1]*multinomial(big(ti-a1-a2),a1,a2)*multinomial(big(tj-b1-b2),b1,b2)*multinomial(big(p-c1-c2),c1,c2)*multinomial(big(tp-d1-d2-d3),d1,d2,d3)*binomial(big(rest),e)*big(q-1)^e*big(q-2)^(a2+b2+c2)*big(q-3)^d3
        else
            coef[j2,t2,p2]=lambda[dist+1]*multinomial(big(ti-a1-a2),a1,a2)*multinomial(big(tj-b1-b2),b1,b2)*multinomial(big(p-c1-c2),c1,c2)*multinomial(big(tp-d1-d2-d3),d1,d2,d3)*binomial(big(rest),e)*big(q-1)^e*big(q-2)^(a2+b2+c2)*big(q-3)^d3
        end
    end         
	return coef;
end

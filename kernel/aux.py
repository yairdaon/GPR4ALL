'''
Created on Apr 29, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
from numpy.linalg import norm
from numpy import zeros
from numpy import array
from numpy import einsum
from numpy import ravel

from math import exp

def cov(x,y, r, d=1):
    '''
    calculate autocovariance as
    k(x,y) = exp(   -||x-y||^2 / r   )
    :param x:
        first point in space
    :param y: 
        second point in space
    :param r:
        the characteristic length scale of
        of the covariance
    :param d:
        the covariance for two points that
        are arbitrarily far apart
        
    * ``cov`` - the covariance between x,y with 
        characteristic length r
    '''
    dist = norm(x-y)/r    
#     t = dist*1.7320508075688772 # sqrt(3)
# 
#     return (1 + t)*math.exp(-t)

    return  d*exp(  -(dist*dist)/2  ) 

def cov_mat(X,r,d=1): 
    '''
    create and return the covariance matrix for the observations
    :param X: 
        a list of locations in space, for which we calculate covariance
    :param r:
        characteristic length scale
    :param d: the covariance for two points that
        are arbitrarily far apart
    '''
    
    #decide on the size of
    n = len(X)
    
    # allocate memory
    C = zeros( (n,n) )
    
    # set the values of the covariance, exploiting symmetry
    for i in range(n):
        for j in range (i):
            C[i,j] = cov(X[i],X[j],r,d)
            C[j,i] = C[i,j]
        C[i,i] = cov(X[i], X[i], r, d)
    return C

def cov_vec(X,w,r,d=1):
    '''
    creates a vector of covariances, between 
    X(a list of numpy arrays) and w (a numpy
    array of same size).
    :param X: 
        a list of locations
    :param w:
        a specific location for which we calculate covariances
    :param r: 
        characteristic length scale
    :param d: the covariance for two points that
        are arbitrarily far apart
        
    returns a vector v s.t. v_i =cov(x_i,w)
    '''
    
    return array( 
#                     [cov(x,w,r,d) for x in X]
                    map(lambda x: cov(x,w,r,d) , X )
                    )

      
def tychonoff_solver( specs, b ):
    '''
    solve Ax = b  using tychonoff regularization or, 
    equvalently, multiply A^{-1}b stably.
    we return the solution to the optimization problem
    x = argmin ||Ax -b||^2 + reg*||x||^2.
    U, S, V is the SVD of A ( A = USV and V is NOT transposed!! )
    reg is the regularization coefficient
    :param U,S,V: the SVD of the matrix A. It holds that
        U*S*V = A (V and not V*, this is what python's
        numpy.linalg.svd(A) returns)
    :param b:
        the vector for which we solve
    * ``x`` - solution to the abovementioned optimization problem
    '''
    S = specs.S
    b = ravel(b)
    c = einsum( ' ij ,  i -> j' , specs.U, b)  # c = U^t * b
    y = c*S/(S*S + specs.reg )            # y_i  = s_i * c_i  /  (s_i^2 + reg
    x = einsum( 'ij , i -> j', specs.V ,  y )  # x = V^t * y

    return x




# 
# def rosenbrock_KL_MC(specs ,tol = 1E-4  ):
#     '''
#     a function that calculates the KL divergence between a 
#     the exp(- rosenbrock ) probability distribution and a 
#     kriged probability distribution
#     '''
#      
#     # we calculated this  analytically
# #     roseNormalization = math.pi/10
#     nwalkers=100
#     burn=200
#     nsteps = burn*15
#      
#     # the initial set of positions
#     pos = np.random.rand(2 * nwalkers) #choose U[0,1]
#     pos = ( 2*pos  - 1.0 ) # shift and stretch
#     pos = pos.reshape((nwalkers, 2)) # reshape
#      
# #     # set the initial state of the PRNG
# #     state = np.random.get_state()
#      
#     # create the emcee sampler and let it burn in
#     sam = mc.EnsembleSampler(nwalkers, 2, truth.rosenbrock_2D)
#      
#     # burn in and then run the sampler
#     pos , _ , _  = sam.run_mcmc(pos, burn)
#     sam.reset()
#     sam.run_mcmc(pos, nsteps)
#      
#     # get the chain
#     sam.sample( nsteps )
#     chain = sam.flatchain
#     lnProbChain = sam.flatlnprobability
#     kullbackLiebler = 0
#     krigNormalization = 0
#     for i in range(nwalkers*nsteps):
#          
#         # the position
#         w = chain[ i , : ]
#          
#         # coresponding kriged LL
#         krig = kg.kriging(w, specs)[0]
#          
#         # the true negative rosenbrock log-likelihood
#         rose = lnProbChain[i]
#          
#         kullbackLiebler  +=   rose  - krig
#         krigNormalization += math.exp( krig - rose ) #since it is the negative of rosenbrock  
#              
#     kullbackLiebler = kullbackLiebler/(nwalkers*nsteps)   
#     krigNormalization = krigNormalization/(nwalkers*nsteps)     
#      
#     integratedTime = mc.autocorr.integrated_time(x, axis)
#     std = /nsteps
#     return kullbackLiebler +  math.log( krigNormalization )

#         
# 
# def rosenbrock_KL_quad(specs ,tol = 1E-4  ):
#     M = 5
#     roseNormalization = math.pi/10
#     def krig( x,y , specs):
#         return math.exp(kg.kriging(np.array( [ x,y ]), specs)[0])
#       
#     Z = scipy.integrate.dblquad(krig, -M, M, lambda x: -M, lambda x: M, args=(specs,) )[0]
#      
#     
#     def integrand(x , y ,specs): 
#         p = np.array( [x,y])
#         rose = truth.rosenbrock_2D(p)
#         krig = kg.kriging(p, specs)[0]
#         arg = math.exp( rose )*( rose - krig )  
#         return arg 
#         
#     first = scipy.integrate.dblquad(integrand, -M, M, lambda x: -M, lambda x: M, args=(specs,) )[0]
#     return first/roseNormalization + math.log(Z/roseNormalization)
module eigensolvers
using Base.Test
import Base.dot
using PyPlot

# A,B and P must implement *
# A must be hermitian, A implemented as a function (not a matrix)
# B and P must be HDP
Id(x) = x
function eig_pcg(A,x0;B=Id,P=Id,tol=1e-16,maxiter=400,do_cg=true)
    #pre-allocate big arrays
    X = zeros(eltype(x0),size(x0)[1],3)
    AX = zeros(eltype(x0),size(x0)[1],3)
    BX = zeros(eltype(x0),size(x0)[1],3)

    # ensure normalization of x0
    Bx0 = B(x0)
    x = (1 / sqrt(real(dot(x0,Bx0))))*x0
    Ax = A(x)
    Bx = Bx0 / sqrt(real(dot(x0,Bx0)))
    @test isapprox(dot(x,Bx),1)
    lambda = 0
    x_prev = zeros(x0)
    Ax_prev = zeros(x0)
    Bx_prev = zeros(x0)
    res_history = zeros(maxiter)

    for i=1:maxiter
        lambda_prev = lambda
        # build new (preconditioned) residual and convergence test
        lambda = dot(x,Ax) / dot(x,Bx)
        @test abs(imag(lambda)) < 1e-9
        lambda = real(lambda)
        resid = Ax - lambda*Bx
        println("iter $i, λ = $lambda, res = $(norm(resid))")
        res_history[i] = norm(resid)
        if norm(resid) < sqrt(tol) || abs(lambda - lambda_prev) < tol
            return x,lambda, res_history[1:i]
        end
        # P_resid = P(resid)
        # AP_resid = A(P_resid)
        # BP_resid = B(P_resid)

        nsub = (do_cg && (i > 1)) ? 3 : 2

        # build subspace
        X[:,1] = x
        AX[:,1]= Ax
        BX[:,1]= Bx
        X[:,2] = P(resid)
        AX[:,2] = A(X[:,2])
        BX[:,2] = B(X[:,2])
        if nsub == 3
            X[:,3] .= x.-x_prev
            AX[:,3] .= Ax.-Ax_prev
            BX[:,3] .= Bx-Bx_prev
        end

        # bookkeeping previous x
        x_prev = x
        Ax_prev = Ax
        Bx_prev = Bx

        # rotate into new x
        @views begin
            c = rayleigh_ritz(X[:,1:nsub],AX[:,1:nsub],BX[:,1:nsub])
            x = X[:,1:nsub]*c
            Ax = AX[:,1:nsub]*c
            Bx = BX[:,1:nsub]*c
        end
        @test isapprox(dot(x,Bx),1)
    end

    return x,lambda, res_history
end

# Returns a column vector c such that X c is B-normalized and achieves the minimum of the Rayleigh quotient in the subspace spanned by X
function rayleigh_ritz(X, AX, BX)
    hamiltonian = X'*AX
    overlap = X'*BX
    @test isapprox(hamiltonian,hamiltonian',atol=1e-9)
    @test isapprox(overlap, overlap',atol=1e-9)
    hamiltonian = (hamiltonian + hamiltonian')/2
    overlap = (overlap + overlap')/2

    # ## Simpler and faster code, but fails if overlap is badly conditioned
    # D,V = eig(hamiltonian,overlap)
    # c = V[:,1]


    #TOFIX this is still not good enough, and accuracy saturates at 1e-7...
    #A x = lambda B x. Assume B = V S V^* = (V S^1/2) (V S^1/2)^*, then the problem is equivalent to S^-1/2 V^* A V S^-1/2 y = lambda y, with y = S^1/2 V^* x. Here we also allow dropping columns of B in case of ill-conditioning
    S,V = eig(overlap)
    n = count(x -> (x < 1e-16),S)
    V = V[:,n+1:end]
    S = diagm(S[n+1:end]) # could be optimized
    ham_reduced = S^(-1/2)*V'*hamiltonian*V*S^(-1/2)
    ham_reduced = (ham_reduced + ham_reduced')/2
    D,U = eig(ham_reduced)
    c = U[:,1]
    @test isapprox(norm(c),1)
    c = V*(S^(-1/2)*c)


    @test isapprox(dot(c,overlap*c),1)
    c = c / sqrt(real(dot(c,overlap*c))) # ensure normalization, normally not needed if eigensolver recognizes ham and ovl are hermitian
    return c
end

function test_pcg()
    srand(0)
    n = 500
    A = randn(n,n) + im*randn(n,n)
    A = (A+A')/2 # force hermitian
    B = randn(n,n) + im*randn(n,n)
    B = (B'*B) + I # force HDP
    x0 = ones(Complex128,n)
    x,lambda, res_history = eig_pcg(x->A*x,x0,B=(x->B*x), maxiter=1500)
    println(lambda,eig(A,B)[1][1])
    semilogy(res_history)
    @test isapprox(lambda,eig(A,B)[1][1],rtol=1e-6)
end
end

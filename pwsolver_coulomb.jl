"""
PW solver with Coulomb potential separated into two parts : radial + regular
"""

module pw_coulomb
using Base.Test
using eigensolvers
using QuadGK
using Combinatorics
# import Base.* #tests for iterative eigensolver
# import Base.transpose
# import Base.size
# import Base.eltype
# import Base.LinAlg.issymmetric
# import Base.LinAlg.A_mul_B!



"""
Type params contain all info for PW calculations
   """

   type params
      N1 :: Int #number of plane-waves in direction 1
      N2 :: Int
      N3 :: Int
      Ntot :: Int #(2N1+1)(2N2+1)(2N3+1)
      size_psi :: Tuple{Int,Int,Int} #(2N1+1,2N2+1,2N3+1)
      L1 :: Float64 #boxsize
      L2 :: Float64
      L3 :: Float64
      V :: Function #potential
      V_grid :: Array{Complex128,3} #Fourier coefficients of V
      kin :: Array{Float64,3} #kinetic operator in Fourier
      fft_plan :: Base.DFT.FFTW.cFFTWPlan{Complex{Float64},-1,false,3} #typeof(plan_fft(zeros(Complex128,(2,2,2))))

      function params(N,L,X,Z)
         function V(x,y,z)
            out = 0.
            for i in 1:size(X,2)
               out -= Z/norm([x,y,z]-X[:,i])
            end
            return out
         end
         p = new(N,N,N,(2N+1)^3,(2N+1,2N+1,2N+1),L,L,L,V)
         p.V_grid = zeros(Complex128,(2N+1,2N+1,2N+1))
         for i in 1:size(X,2)
            p.V_grid += Z*coulomb(p,X[:,i])
         end
         p.kin = zeros(2N+1,2N+1,2N+1)
         for i3 in 1:(2*p.N3+1)
            for i2 in 1:(2*p.N2+1)
               for i1 in 1:(2*p.N1+1)
                  p.kin[i1,i2,i3] = 2*pi^2*((dif(i1,p.N1)/p.L1)^2 + (dif(i2,p.N2)/p.L2)^2
                  + (dif(i3,p.N3)/p.L3)^2)
               end
            end
         end
         p.fft_plan = plan_fft(zeros(Complex128,p.size_psi))
         return p
      end
   end

   function multiplier(N)
      return max(1,trunc(Int,100/N)) :: Integer
   end

   #Coulomb potential separated into two parts : radial + regular
   function coulomb(p :: pw_coulomb.params,X)
      mult = multiplier(p.N1)
      function chi(r)
         if r > 1
            return 0
         else
            return exp(-r^4/(1-r^2))
         end
      end
      V_rad = zeros(Complex128, (2p.N1+1,2p.N1+1,2p.N1+1))
      V_fft = zeros(Complex128, (2mult*p.N1+1,2mult*p.N1+1,2mult*p.N1+1))
      #radial part
      for i3 in 1:2p.N1+1
         for i2 in 1:i3
            for i1 in 1:i2
               k_vec = fft_mode.([i1,i2,i3],[p.N1,p.N2,p.N3])
               k = norm(k_vec)
               integrand(r) = chi(r)*sin(2pi*k*r/p.L1)
               if k==0.
                  V_rad[i1,i2,i3] = -4pi/(p.L1^3)*QuadGK.quadgk(r -> chi(r)*r,0,1)[1]
               else
                  V_rad[i1,i2,i3] = -2/(k*p.L1^2)*QuadGK.quadgk(r -> chi(r)*sin(2pi*k*r/p.L1), 0, 1)[1]
                  #remplissage du reste de la matrice
                  P = unique(permutations([i1,i2,i3]))
                  for j in 2:endof(P)
                     k_vec = fft_mode.(P[j],[p.N1,p.N1,p.N1])
                     V_rad[P[j]...] = exp(-2*im*pi*dot(k_vec,X)/p.L1)*V_rad[i1,i2,i3]
                  end
                  k_vec = fft_mode.([i1,i2,i3],[p.N1,p.N1,p.N1])
                  V_rad[i1,i2,i3] = exp(-2*im*pi*dot(k_vec,X)/p.L1)*V_rad[i1,i2,i3]
               end
            end
         end
      end
      #FFT
      for i3 in 1:2mult*p.N3+1
         for i2 in 1:2mult*p.N2+1
            for i1 in 1:2mult*p.N1+1
               r = norm([(i1-1)/(2mult*p.N1+1)*p.L1, (i2-1)/(2mult*p.N2+1)*p.L2, (i3-1)/(2mult*p.N3+1)*p.L3] - X)
               V_fft[i1,i2,i3] =  -(1-chi(r))/r
            end
         end
      end
      return p.Ntot*V_rad + fft_reshape(fft(V_fft),p,mult)
   end

   """
   FFT stuff
   """
   function dif(i,N)
      if (i <= N+1)
         return i-1
      else
         return i-2*N-2
      end
   end

   function coords(p::params,i1,i2,i3)
      return [(i1-1)/(2*p.N1+1)*p.L1, (i2-1)/(2*p.N2+1)*p.L2, (i3-1)/(2*p.N3+1)*p.L3]
   end

   function fft_mode(i,N)
      if (i <= N+1)
         return i-1
      else
         return i-2*N-2
      end
   end

   function fft_reshape(A, p::pw_coulomb.params, mult) #uglier than with fftshift but faster
      B = zeros(Complex128, p.size_psi)
      function aux(i)
         if i==0
            return 1:(p.N1+1)
         else
            return (p.N1*(2mult-1)+2):(2p.N1*mult+1)
         end
      end
      function fft_mode_vec(i)
         if i==0
            return 1:(p.N1+1)
         else
            return (p.N1+2):(2p.N1+1)
         end
      end
      for i1 in 0:1
         for i2 in 0:1
            for i3 in 0:1
               B[fft_mode_vec(i1), fft_mode_vec(i2), fft_mode_vec(i3)] = A[aux(i1), aux(i2), aux(i3)]
            end
         end
      end
      return p.Ntot/((2*p.N1*mult+1)*(2*p.N2*mult+1)*(2*p.N3*mult+1))*B
   end



   # Applies H on psi
   function ham(p::params, psi)
      @assert size(psi) == p.size_psi
      return p.fft_plan*((p.fft_plan\psi).*(p.fft_plan\p.V_grid)) .+ p.kin.*psi
   end

   # function *(p::params,psi::Array{Complex128,1})
   #    return H(psi)
   # end
   # function size(p::params)
   #    return (p.Ntot,p.Ntot)
   # end
   # function eltype(p::params)
   #    return eltype(1.2*im)
   # end
   #
   # function Base.LinAlg.issymmetric(p::params)
   #    return true
   # end

   #solves the eigenproblem with psi0
   function energy(p::params, psi0; args...)
      H(psi) = reshape(ham(p, reshape(psi,p.size_psi)),p.Ntot)
      function P(psi)
         meankin = sum(p.kin[i]*abs2(psi[i]) for i = 1:p.Ntot) / (vecnorm(psi)^2)
         return psi ./ (.2*meankin .+ p.kin[:]) # this should be tuned but works more or less
      end

      #return eigs(p,v0=psi0[:],nev=3, which=:SR)
      return eigensolvers.eig_pcg(H, psi0[:], P=P; args...)
   end


end



module tests_pwcoulomb
using pw_coulomb
#using FastConv

# Solution of the hydrogen atom centered at X
function phi_H(p::pw_coulomb.params, X, Z)
   function phi_H_real(x,y,z,X)
      r = sqrt((x-X[1])^2 +(y-X[2])^2+(z-X[3])^2)
      return exp(-Z*r)
   end
   mat = zeros(Complex128,p.size_psi)
   for i3 in 1:(2*p.N3+1)
      for i2 in 1:(2*p.N2+1)
         for i1 in 1:(2*p.N1+1)
            mat[i1,i2,i3] = phi_H_real(pw_coulomb.coords(p,i1,i2,i3)..., X)
         end
      end
   end
   return p.fft_plan*mat
end

function phi_H2(p::pw_coulomb.params, X1, X2, Z)
   return phi_H(p,X1,Z) + phi_H(p,X2,Z)
end

# Answer for infinite N,L: -0.5
function H_test(N,L,Z)
   X = zeros(3,1)
   X[:,1] = [L/2,L/2,L/2]
   p = pw_coulomb.params(N,L,X,Z)
   psi, E, res = pw_coulomb.energy(p,phi_H(p,[L/2,L/2,L/2],Z),
   tol=1e-4, maxiter=400)
   return psi,E
end



# X1, X2: coordinates of the atoms
function V_H2(x,y,z, X1, X2)
   return -1./sqrt((x-X1[1])^2+(y-X1[2])^2+(z-X1[3])^2) - 1./sqrt((x-X2[1])^2+(y-X2[2])^2+(z-X2[3])^2)
end

function H2_test(N,L,R,Z)
   X1 = [(L+R)/2,L/2,L/2]
   X2 = [(L-R)/2,L/2,L/2]
   X = zeros(3,2)
   X[:,1] = X1
   X[:,2] = X2
   p = pw_coulomb.params(N,L,X,Z)
   psi, E, res = pw_coulomb.energy(p,phi_H2(p,X1,X2,Z),
   tol=1e-10, maxiter=400)
   return psi, E
end


function conv_H2(X1, X2, Nmin, Nmax, L)
   Ns = Nmin:5:Nmax
   Es = zeros(length(Ns))
   Psi = Array(Any, length(Ns))
   for (i,N) in enumerate(Ns)
      println(n)
      psi, E, res = H2_test(N,L,X1,X2)
      Es[i] = E
      # Psi[Integer(round((n-Nmin)/5))+1] = psi
   end
   writedlm("Direct_Energy_45.txt", Es)
   # writedlm("Direct_Psi_45.txt", Psi)
end

function herm(N,L,Z)
   V(x,y,z) = 0.
   p = pw_coulomb.params(N,L,V)
   psi = phi_H(p, [L/2,L/2,L/2], Z)
   pot = zeros(Complex128, (4p.N1+1,4p.N2+1,4p.N3+1))
   for i1 in 4p.N1+1
      for i2 in 4p.N2+1
         for i3 in 4p.N3+1
            k_vec = pw_coulomb.fft_mode.([i1,i2,i3],[p.N1,p.N2,p.N3])
            if k_vec == [0,0,0]
               pot[i1,i2,i3] = complex(0.)
            else
               pot[i1,i2,i3] += - p.L1^2*exp(-2*im*pi*dot(k_vec,[L/2,L/2,L/2])/p.L1)/pi/dot(k_vec,k_vec)
            end
         end
      end
   end
   pot_psi = convn(fftshift(pot),fftshift(psi))
   return vecdot(psi,ifftshift(pot_psi[2p.N1+1:4p.N1+1,2p.N2+1:4p.N2+1,2p.N3+1:4p.N3+1]))
end

# #test for V = -Z/|x| and check if the solution is radial
# function energy_check(V, N1, N2, N3, L1, L2, L3, seed)
#   psi, E, res = energy(V, N1, N2, N3, L1, L2, L3, seed)
#   psi = reshape(psi, (2*N1+1, 2*N2+1, 2*N3+1))
#   ifft!(psi)
#   psi = reshape(psi, (2*N1+1)*(2*N2+1)*(2*N3+1))
#   x = Array(Any, (2*N1+1,2*N2+1,2*N3+1))
#   for i3 in 1:(2*N1+1)
#     for i2 in 1:(2*N2+1)
#       for i1 in 1:(2*N3+1)
#         x[i1,i2,i3] = sqrt(((i1-1)/(2*N1+1)*L1 - L1/2)^2 +
#          ((i2-1)/(2*N2+1)*L2 - L2/2)^2 + ((i3-1)/(2*N3+1)*L3 - L3/2)^2  )
#       end
#     end
#   end
#   return reshape(x, (2*N1+1)*(2*N2+1)*(2*N3+1)), psi
# end

end

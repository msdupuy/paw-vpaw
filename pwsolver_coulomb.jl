"""
PW solver with Coulomb potential separated into two parts : radial + regular
Coulomb potential Fourier coefficients computed on a grid twice as large because of convolution with \hat\psi_K
"""

module pw_coulomb
using Base.Test
using eigensolvers
using QuadGK
using Combinatorics

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

      function params(N,L,X,Z,V)
         # function V(x,y,z)
         #    out = 0.
         #    for i in 1:size(X,2)
         #       out -= Z/norm([x,y,z]-X[:,i])
         #    end
         #    return out
         # end
         p = new(N,N,N,(2N+1)^3,(2N+1,2N+1,2N+1),L,L,L,V)
         p.V_grid = zeros(Complex128,(4N+1,4N+1,4N+1)) #larger grid for convolution
         for i in 1:size(X,2)
            p.V_grid += coulomb(p,X[:,i],V)
         end
         p.kin = zeros(2N+1,2N+1,2N+1)
         for i3 in 1:(2p.N3+1)
            for i2 in 1:(2p.N2+1)
               for i1 in 1:(2p.N1+1)
                  p.kin[i1,i2,i3] = 2*pi^2*((fft_mode(i1,p.N1)/p.L1)^2 + (fft_mode(i2,p.N2)/p.L2)^2
                  + (fft_mode(i3,p.N3)/p.L3)^2)
               end
            end
         end
         p.fft_plan = plan_fft(zeros(Complex128,(4p.N1+1,4p.N2+1,4p.N3+1)))
         return p
      end
   end

   function multiplier(N) #returns FFT multiplier to have consistent FFT with regard to the plane-wave cut-off
      return max(2,trunc(Int,60/(2N))) :: Integer
   end

   #Coulomb potential separated into two parts : radial + regular, computed on a grid twice larger
   #V : radial potential
   function coulomb(p :: pw_coulomb.params,X,V)
      mult = multiplier(2p.N1)
      function chi(r)
         if r > 1.
            return 0.
         else
            return exp(-r^6/(1-r^6))
         end
      end
      V_rad = zeros(Complex128, (4p.N1+1,4p.N1+1,4p.N1+1))
      V_fft = zeros(Complex128, (4mult*p.N1+1,4mult*p.N1+1,4mult*p.N1+1))
      #radial part
      for i3 in 1:4p.N1+1
         for i2 in 1:i3
            for i1 in 1:i2
               k_vec = fft_mode.([i1,i2,i3],[2p.N1,2p.N2,2p.N3])
               k = norm(k_vec) :: Float64
               let k=k #otherwise type not inferred...
               integrand(r) = chi(r)*sin(2pi*k*r/p.L1)
                  Base.Test.@inferred integrand(1.0)
                  if k==0.
                     V_rad[i1,i2,i3] = 4pi/(p.L1^3)*QuadGK.quadgk(r -> V(r)*chi(r)*r^2,0,1)[1]
                  else
                     V_rad[i1,i2,i3] = 2/(k*p.L1^2)*QuadGK.quadgk(r -> V(r)*chi(r)*sin(2pi*k*r/p.L1)*r, 0, 1, abstol=1e-8, order=10)[1]
                     #remplissage du reste de la matrice
                     P = unique(permutations([i1,i2,i3]))
                     for j in 2:endof(P)
                        k_vec = fft_mode.(P[j],[2p.N1,2p.N1,2p.N1])
                        V_rad[P[j]...] = exp(-2*im*pi*dot(k_vec,X)/p.L1)*V_rad[i1,i2,i3]
                     end
                     k_vec = fft_mode.([i1,i2,i3],[2p.N1,2p.N1,2p.N1])
                     V_rad[i1,i2,i3] = exp(-2*im*pi*dot(k_vec,X)/p.L1)*V_rad[i1,i2,i3]
                  end
               end
            end
         end
      end
      #FFT
      for i3 in 1:4mult*p.N3+1
         for i2 in 1:4mult*p.N2+1
            for i1 in 1:4mult*p.N1+1
               r = norm(coords(p,i1,i2,i3,2mult) - X)
               V_fft[i1,i2,i3] =  (1-chi(r))*V(r)
            end
         end
      end
      return (4p.N1+1)^3*V_rad + fft_reshape(fft(V_fft),p,mult,2)
   end

   """
   FFT stuff
   """
   function coords(p::params,i1,i2,i3,mult)
      return [(i1-1)/(2mult*p.N1+1)*p.L1, (i2-1)/(2mult*p.N2+1)*p.L2, (i3-1)/(2mult*p.N3+1)*p.L3]
   end

   function fft_mode(i,N)
      if (i <= N+1)
         return i-1
      else
         return i-2*N-2
      end
   end

   function aux(i,p::params,mult,n)
      if i==0
         return 1:(n*p.N1+1) #positive frequencies
      else
         return (n*p.N1*(2mult-1)+2):(2*n*p.N1*mult+1) #negative frequencies
      end
   end

   function fft_mode_vec(i,p::params,n)
      if i==0
         return 1:(n*p.N1+1) #positive frequencies
      else
         return (n*p.N1+2):(2n*p.N1+1) #negative frequencies
      end
   end

   function fft_reshape(A, p::pw_coulomb.params, mult,n) #uglier than with fftshift but faster
      B = zeros(Complex128, (2n*p.N1+1,2n*p.N2+1,2n*p.N3+1))
      for i1 in 0:1
         for i2 in 0:1
            for i3 in 0:1
               B[fft_mode_vec(i1,p,n), fft_mode_vec(i2,p,n), fft_mode_vec(i3,p,n)] = A[aux(i1,p,mult,n), aux(i2,p,mult,n), aux(i3,p,mult,n)]
            end
         end
      end
      return (2n*p.N1+1)*(2n*p.N2+1)*(2n*p.N3+1)/((2n*p.N1*mult+1)*(2n*p.N2*mult+1)*(2n*p.N3*mult+1))*B
   end

   function test_reshape() #missing fft multiplier
      X = zeros(3,1)
      X[:,1] = [2.5,2.5,2.5]
      p = params(5,5.,X,3.)
      psi = randn(21,21,21)
      return vecnorm(fft_reshape(psi,p,2) - ifftshift(fftshift(psi)[p.N1+1:3p.N1+1,p.N1+1:3p.N1+1,p.N1+1:3p.N1+1]))
   end

   function inv_fft_reshape(psi, p::params)
      fft_large_psi = zeros(Complex128,(4p.N1+1,4p.N2+1,4p.N3+1))
      for i1 in 0:1
         for i2 in 0:1
            for i3 in 0:1
               fft_large_psi[aux(i1,p,2,1), aux(i2,p,2,1), aux(i3,p,2,1)] = psi[fft_mode_vec(i1,p,1), fft_mode_vec(i2,p,1), fft_mode_vec(i3,p,1)]
            end
         end
      end
      return ((4p.N1+1)*(4p.N2+1)*(4p.N3+1))/p.Ntot*fft_large_psi
   end




   # Applies H on psi
   function ham(p::params, psi)
      @assert size(psi) == p.size_psi
      return fft_reshape(p.fft_plan*((p.fft_plan\p.V_grid).*(p.fft_plan\inv_fft_reshape(psi,p))), p, 2, 1) .+ p.kin.*psi
   end

   #solves the eigenproblem with psi0
   function energy(p::params, psi0; args...)
      H(psi) = reshape(ham(p, reshape(psi,p.size_psi)),p.Ntot)
      function P(psi)
         meankin = sum(p.kin[i]*abs2(psi[i]) for i = 1:p.Ntot) / (vecnorm(psi)^2)
         return psi ./ (.2*meankin .+ p.kin[:]) # this should be tuned but works more or less
      end
#      return eigensolvers.eig_lanczos(H, psi0[:], m=5, Imax = 1000)
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
            mat[i1,i2,i3] = phi_H_real(pw_coulomb.coords(p,i1,i2,i3,1)..., X)
         end
      end
   end
   return fft(mat)
end

function phi_H2(p::pw_coulomb.params, X1, X2, Z)
   return phi_H(p,X1,Z) + phi_H(p,X2,Z)
end

# Answer for infinite N,L and Z = 1: -0.5,
function H_test(N,L,Z)
   X = zeros(3,1)
   X[:,1] = [L/2,L/2,L/2]
   V(r) = 1./r
   p = pw_coulomb.params(N,L,X,Z,V)
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
   V(r) = -Z/r
   p = pw_coulomb.params(N,L,X,Z,V)
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


   function test_V_symmetric()
      X = zeros(3,1)
      X[:,1] = [2.5,2.5,2.5]
      p = params(5,5.,X,3.)
      p.V_grid = fftshift(p.V_grid)
      V_transpose = zeros(Complex128,(4p.N1+1,4p.N1+1,4p.N1+1))
      for i1 in 1:4p.N1+1
         for i2 in 1:4p.N2+1
            for i3 in 1:4p.N3+1
               V_transpose[i1,i2,i3] = p.V_grid[4p.N1+2-i1,4p.N2+2-i2,4p.N3+2-i3]
            end
         end
      end
      return vecnorm(p.V_grid - conj.(V_transpose),Inf)
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

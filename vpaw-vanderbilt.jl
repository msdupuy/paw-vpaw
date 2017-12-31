module paw

using Polynomials
using QuadGK
using GSL
using Base.Test

#harmonique spherique
function Y_lm(x,y,z,l,m)
   r = sqrt(x^2 + y^2 + z^2)
   phi = atan(y/x)
   if m < 0
      return (-1)^m*sqrt(2)*sin(abs(m)*phi)*GSL.sf_legendre_sphPlm(l, abs(m), z/r)
   elseif m == 0
      return GSL.sf_legendre_sphPlm(l, 0, z/r)
   else # m>0
      return (-1)^m*sqrt(2)*cos(m*phi)*GSL.sf_legendre_sphPlm(l, m, z/r)
   end
end

function Y_lmcomplex(x,y,z,l,m)
   r = sqrt(x^2 + y^2 + z^2)
   phi = atan2(y,x)
   if m >= 0
      return (-1)^m*GSL.sf_legendre_sphPlm(l, m, z/r)*exp(im*m*phi)
   else
      return sqrt((2l+1)/(4pi)*factorial(l+m)/factorial(l-m))*GSL.sf_legendre_Plm(l, -m, z/r)*exp(im*m*phi)
   end
end

#Atomic eigenvalues and normalization factor
function C_nl(n,l,Z)
   return sqrt((2Z/n)^(2l+3)* factorial(n-l-1)/(2*n*factorial(n+l)) )
end

function E_n(n,Z)
   return -0.5*(Z/n)^2
end

#Atomic wave functions

function R_nl(r,n,l,Z)
   if r < 0.
      return 0.
   else
      return C_nl(n,l,Z)*r^(l+1)*GSL.sf_laguerre_n(n-l-1, 2l+1, 2Z*r/n)*exp(-Z*r/n)
   end
end

#Generation of Troullier-Martins pseudopotential
"""
\tilde{R}(r) = r^{l+1} * exp(p(r))
p(r) = \sum_{k=0}^6 c_{2k} r^{2k}
resolution par Newton sur le vecteur rescaled [c_{2k}r_c^{2k}]

coef_p retourne le vecteur des coefficients [c_{2k}]
"""
function coef_TM(rc, n, l, Z, tol)
   #\int_0^1 x^{2k+2l+2} exp(2p(x))
   function NC(coef, k)
      function p(x)
         return x^(2k+2l+2)*exp(2polyval(Poly(coef),x^2))
      end
      return QuadGK.quadgk(p, 0, 1)[1]
   end

   b = [0. for i in 1:7]
   #remplissage du second membre
   function lag_der(k) #derivative of the associated Laguerre polynomials
      if k > n-l-1
         return 0.
      else
         return (-2Z*rc/n)^k*GSL.sf_laguerre_n(n-l-1-k, k+2l+1, 2Z*rc/n)
      end
   end
   L(r) = GSL.sf_laguerre_n(n-l-1, 2l+1, 2Z*r/n)
   b[1] = 1./(rc^(2*l+3.))*QuadGK.quadgk(r -> (r^(l+1)*L(r)*exp(-Z*r/n))^2, 0, rc)[1]
   b[2] = -Z*rc/n + log(L(rc))
   b[3] = -Z*rc/n + lag_der(1)/L(rc)
   b[4] = (L(rc)*lag_der(2) - lag_der(1)^2)/L(rc)^2
   b[5] = (L(rc)^2*lag_der(3) + 2lag_der(1)^3 - 3L(rc)*lag_der(1)*lag_der(2))/L(rc)^3
   b[6] = (-6lag_der(1)^4 + L(rc)^2*(L(rc)*lag_der(4) -3lag_der(2)^2 -4lag_der(3)*lag_der(1)) +12L(rc)*lag_der(1)^2*lag_der(2) )/L(rc)^4

   #methode de Newton sur f(x)-b
   function f(x)
      Q = Poly(x)
      P = Q(Poly([0., 0., 1.]))
      return [NC(x, 0), P(1.), polyder(P)(1.), polyder(P, 2)(1.),
      polyder(P, 3)(1.), polyder(P, 4)(1.), x[2]^2 + (2*l+5)*x[3]]
   end

   function Df(x)
      a1 = zeros(1,7)
      for k in 1:7
         a1[1,k] = 2*NC(x, k-1)
      end
      A = ones(5,7)
      for j in 1:7
         A[2,j] = 2*(j-1)
         A[3,j] = 2*(j-1)*(2*(j-1)-1)
         A[4,j] = 2*(j-1)*(2*(j-1)-1)*(2*(j-1)-2)
         A[5,j] = 2*(j-1)*(2*(j-1)-1)*(2*(j-1)-2)*(2*(j-1)-3)
      end
      a3 = zeros(1,7)
      a3[1,2] = 2*x[2]
      a3[1,3] = 2*l+5
      return [a1 ; A ; a3]
   end

   Imax = 500
   x = -ones(7)
   i = 0
   while (norm(f(x) - b) > tol) & (i < Imax)
      i = i+1
      x = x - inv(Df(x))*(f(x) -b)
   end
   err = norm(f(x)-b)
   #rescaling step
   x = [x[k]/rc^(2*(k-1)) for k in 1:7]
   @test i < Imax
   return [x, i, err]
end

"""
R_V_PP is the radial pseudo wave function of the Troullier-Martins
pseudopotential
p : polynomial in TM pseudo wave function
"""

function R_V_PP(r, rc, n, l, p, Z)
   if r > rc
      return R_nl(r,n,l,Z)
   else
      P = Poly(p)
      return C_nl(n,l,Z)*r^(l+1)*exp(polyval(P,r^2))
   end
end

"""
V_scr is the screened Troullier-Martins pseudopotential
p is the vector of coefficients given by coef_p.
"""
function V_scr(r, n, l, rc, p, Z)
   if r > rc
      return -Z/r
   else
      Q = Poly([0.,0.,1.])
      P = Poly(p)(Q)
      return E_n(n,Z) + (l+1)/r*polyval(polyder(P), r) + 1./2.*(
      polyval(polyder(P), r)^2 + polyval(polyder(P, 2), r)  )
   end
end

function coef_V_scr(n, l, Z, rc, p)
   Q = Poly([0.,0.,1.])
   P = Poly(p)(Q)
   return Poly([E_n(n,Z)]) + (l+1)*(polyder(P) ÷ Poly([0, 1.])) + 0.5*(
   polyder(P)*polyder(P) + polyder(P, 2)  )
end

"""
PAW functions Generation (Vanderbilt)
"""

#coef_tilde_R retourne les coefficients de la partie radiale de \tilde{R}_{nl}
function coef_tilde_R(rc, n, l, Z)
   A = ones(5,5)
   for j in 1:5
      A[2,j] = 2*(j-1)
      A[3,j] = 2*(j-1)*(2*(j-1)-1)
      A[4,j] = 2*(j-1)*(2*(j-1)-1)*(2*(j-1)-2)
      A[5,j] = 2*(j-1)*(2*(j-1)-1)*(2*(j-1)-2)*(2*(j-1)-3)
   end
   b = zeros(5)
   L(r) = GSL.sf_laguerre_n(n-l-1, 2l+1, 2Z*r/n)
   function lag_der(k) #derivative of the associated Laguerre polynomials
      if k > n-l-1
         return 0.
      else
         return (-1)^k*GSL.sf_laguerre_n(n-l-1-k, k+2l+1, 2Z*rc/n)
      end
   end
   b[1] = L(rc)*exp(-Z*rc/n)
   b[2] = Z*rc/n*exp(-Z*rc/n)*(2*lag_der(1)-L(rc))
   b[3] = (Z*rc/n)^2*exp(-Z*rc/n)*(4*lag_der(2)-4lag_der(1)+L(rc))
   b[4] = (Z*rc/n)^3*exp(-Z*rc/n)*(8*lag_der(3)-12lag_der(2)+6lag_der(1)-L(rc))
   b[5] = (Z*rc/n)^4*exp(-Z*rc/n)*(16*lag_der(4)-32lag_der(3)+24lag_der(2)-8lag_der(1)+L(rc))
   x = \(A,b)
   return [x[k]/rc^(2*(k-1)) for k in 1:5]
end

"""
Radial PAW pseudo wave function
"""

function tilde_R_nl(r, rc, n, l, Z, coef_PAW)
   if r < rc
      return C_nl(n,l,Z)*r^(l+1)*polyval(Poly(coef_PAW), r^2)
   else
      return R_nl(r, n, l, Z)
   end
end

"""
auxiliary functions for projector functions
"""

function chi_nl(r, rc, n, l, Z, coef_PAW, coef_PP)
   if r > rc
      return 0.
   else
      Q = Poly([0.,0.,1.])
      P = Poly(coef_PAW)(Q)
      return r^l*(0.5*r*polyval(polyder(P, 2), r)
      +(l+1)*polyval(polyder(P), r)
      + r*(E_n(n,Z) - V_scr(r, l+1, l, rc, coef_PP, Z))*polyval(P,r))
   end
end

function coef_chi_nl(rc, n, l, Z, coef_PAW, coef_PP)
   Q = Poly([0.,0.,1.])
   P = Poly(coef_PAW)(Q)
   return ( 0.5*Poly([0,1])*polyder(P, 2) + (l+1)*polyder(P) +
   Poly([0,1])*(Poly([E_n(n,Z)]) - coef_V_scr(l+1, l, Z, rc, coef_PP))*P
   ) * poly(zeros(l))
end

"""
Attention jusqu'a present, coef_PAW et coef_V_PP est une liste de coefficients
pour (n,l) fixe.
Desormais, ce sont des tableaux multidimensionels

GS_VdB calcule la matrice de Gram notee B dans le document de travail
"""

function GS_VdB(N_I, l, Z, rc, coef_PAW, coef_PP)
   B = zeros(N_I,N_I)
   for n in 1:N_I
      for m in 1:N_I
         L_n = coef_PAW[:,n]
         L_m = coef_PAW[:,m]
         L_V = coef_PP
         B[n,m] = QuadGK.quadgk(r -> tilde_R_nl(r, rc, n+l, l, Z, L_n)*
         chi_nl(r, rc, m+l, l, Z, L_m, L_V), 0., rc, abstol=1e-14, reltol=1e-14)[1]
      end
   end
   return B
end

"""
Radial part of the projector functions by Vanderbilt scheme
n : projector number
N : number of s-state PAW functions
"""
function rad_proj(r, B, n, l, Z, rc, coef_PAW, coef_PP, N)
   return (transpose(B)\[chi_nl(r, rc, k+l, l, Z, coef_PAW[:,k], coef_PP) for k in 1:N])[n]
end

function coef_rad_proj(B, l, Z, rc, coef_PAW, coef_PP, N)
   return transpose(inv(B))*[coef_chi_nl(rc, i+l, l, Z, coef_PAW[:,i], coef_PP) for i in 1:N]
end

"""
Projector function by custom scheme
cut-off = (r/rc*(1-r/rc))^2
"""

function GS_custom(N_I, l, Z, rc, coef_PAW, coef_PP)
   B = zeros(N_I,N_I)
   for n in 1:N_I
      for m in 1:N_I
         L_n = coef_PAW[:,n]
         L_m = coef_PAW[:,m]
         L_V = coef_PP
         B[n,m] = QuadGK.quadgk(r -> (r/rc*(1-r/rc))^2*tilde_R_nl(r, rc, n+l, l, Z, L_n)*tilde_R_nl(r, rc, m+l, l, Z, L_m), 0., rc, abstol=1e-14, reltol=1e-14)[1]
      end
   end
   return B
end

function rad_proj_custom(r, B, n, l, Z, rc, coef_PAW, coef_PP, N)
   return (transpose(B)\[(r/rc*(1-r/rc))^2*tilde_R_nl(r, rc, i+l, l, Z, coef_PAW[:,i]) for i in 1:N])[n]
end

function coef_proj_custom(B, l, Z, rc, coef_PAW, coef_PP, N)
   return transpose(inv(B))*[1/rc^4*poly([0,0,rc,rc]) *C_nl(i+l,l,Z)*poly(zeros(l+1))*Poly(coef_PAW[:,i])(poly([0,0])) for i in 1:N]
end

"""
Matrices D_ij and S_ij
"""
function Ddiff(r, rc, n, coef_PAW, l, Z)
   if r > rc
      return 0.
   else
      P = Poly(coef_PAW)
      L(r) = GSL.sf_laguerre_n(n-l-1, 2l+1, 2Z*r/n)
      function lag_der(r) #derivative of the associated Laguerre polynomials
         if 1 > n-l-1
            return 0.
         else
            return (-1)*GSL.sf_laguerre_n(n-l-2, 2+2l, 2Z*r/n)
         end
      end
      der_phi = exp(-Z*r/n)*(-Z/n*r^(l+1)*L(r) + (l+1)*r^l*L(r) + 2Z/n*lag_der(r)*r^(l+1))
      der_tdphi = 2r^(l+2)*polyval(polyder(P),r^2) + (l+1)*r^l*polyval(P,r^2)
      return C_nl(n,l,Z)*(der_phi - der_tdphi)
   end
end

"""
R : distance entre les deux potentiels coulombiens
19/04 : D_ij modified for pawsolver testing
21/04 : tests over D_ij adapted to H_2 potential
02/06 : D_ij obsolète
"""
#number of functions for the PAW range Npaw considered
function Npawtot(Npaw)
    Ntot = 0
    for lpaw in eachindex(Npaw)
      Ntot += Npaw[lpaw]*(2lpaw-1)
   end
   return Ntot
end

function S_ij(rc, Z, coef_PAW, Npaw)
   S = zeros(Npawtot(Npaw),Npawtot(Npaw))
   ind = 0
   Rnl_tdRnl(r,n,l) = R_nl(r, n, l, Z) - tilde_R_nl(r, rc, n, l, Z, coef_PAW[:,n-l,l+1])
   for lpaw in eachindex(Npaw)
      for m in 1:(2lpaw-1)
         for j in 1:Npaw[lpaw]
            for k in 1:Npaw[lpaw]
               S[j+ind,k+ind] = QuadGK.quadgk(r -> Rnl_tdRnl(r,j+lpaw-1,lpaw-1)*Rnl_tdRnl(r,k+lpaw-1,lpaw-1), 0., rc, reltol=1e-10)[1]
            end
         end
         ind += Npaw[lpaw]
      end
   end
   return S
end

type pawcoef
   Z :: Float64 #atomic charge
   rc :: Float64 #cutoff radius
   Npaw :: Array{Integer,1}
   tdR  #\tilde R_nl coefficients
   proj #projectors coefficients
   coef_TM
   function pawcoef(Z, rc :: Float64, Npaw :: Array{Int64,1}; GS = GS_VdB, proj_gen = coef_rad_proj)
      coefpaw = new(Z, rc, Npaw, nothing)
      coefpaw.tdR = zeros(5,max(Npaw...),endof(Npaw))
      coefpaw.proj = Array{Polynomials.Poly{Float64}}(max(Npaw...),endof(Npaw))
      coef_PP = zeros(7,endof(Npaw))
      for lpaw in eachindex(Npaw)
         for i in 1:Npaw[lpaw]
            coefpaw.tdR[:,i,lpaw] = coef_tilde_R(rc, i+lpaw-1, lpaw-1, Z)
         end
         coef_PP[:,lpaw] = coef_TM(rc, lpaw, lpaw-1, Z, 1e-10)[1]
         B = GS(Npaw[lpaw], lpaw-1, Z, rc, coefpaw.tdR[:,:,lpaw], coef_PP[:,lpaw])
         coefpaw.proj[1:Npaw[lpaw],lpaw] = proj_gen(B, lpaw-1, Z, rc, coefpaw.tdR[:,:,lpaw], coef_PP[:,lpaw], Npaw[lpaw])
      end
      coefpaw.coef_TM = coef_TM(rc, 1, 0, Z, 1e-8)[1]
      return coefpaw
   end
end


end



module pawgentest
using Polynomials
using paw
using PyPlot
using Cubature
using QuadGK
using GSL

function int3D(f,rc) #integration on the box [-rc,rc]^3 for functions with singularity at (0,0,0) // awful compared to radial 1D integration
   out = 0
   for i1 in 0:1
      for i2 in 0:1
         for i3 in 0:1
            out += Cubature.hcubature(f, [-rc*(1-i1),-rc*(1-i2),-rc*(1-i3)], [rc*i1,rc*i2,rc*i3],abstol=1e-5,reltol=1e-5, maxevals=100000)[1]
         end
      end
   end
   return out
end

function int_to_nl(ipaw,Npaw)
   l = 0
   while (ipaw > (2l+1)*Npaw[l+1])
      ipaw -= (2l+1)*Npaw[l+1]
      l += 1
   end
   m = -l
   while (ipaw/Npaw[l+1] > 1)
      ipaw -= Npaw[l+1]
      m += 1
   end
   n = ipaw
   return [l+1,n,m]
end

function TM_test(rc, n, l, Z)
   p = paw.coef_TM(rc, n, l, Z, 1e-8)[1]
   plot(0:0.01:2rc, [paw.V_scr(r, n, l, rc, p, Z) for r in 0:0.01:2rc])
end

function RV_PPtest(rc, n, l, Z)
   p = paw.coef_TM(rc, n, l, Z, 1e-8)[1]
   plot(0:0.01:2rc, [paw.R_V_PP(r, rc, n, l, p, Z) for r in 0:0.01:2rc])
   plot(0:0.01:2rc, [paw.R_nl(r, n, l, Z) for r in 0:0.01:2rc])
end

function td_Rnl_test(rc,n,l,Z)
   p = paw.coef_tilde_R(rc, n, l,Z)
   plot(0:0.001:2rc, [paw.tilde_R_nl(r, rc, n, l, Z, p) for r in 0:0.001:2rc],"b-")
   plot(0:0.001:2rc, [paw.R_nl(r, n, l, Z) for r in 0:0.001:2rc],"r-")
end


function proj_test(rc,n,l,Z,N)
   coef_PAW = zeros(5,N)
   for i in 1:N
      coef_PAW[:,i] = paw.coef_tilde_R(rc,i+l,l,Z)
   end
   coef_PP = paw.coef_TM(rc, l+1, l, Z, 1e-8)[1]
   B = paw.GS_VdB(N, l, Z, rc, coef_PAW, coef_PP)
   plot(0:0.01:2rc, [paw.rad_proj(r, B, n-l, l, Z, rc, coef_PAW, coef_PP, N) for r in 0:0.01:2rc])
end

function proj_test_poly(rc,n,l,Z,N)
   coef_PAW = zeros(5,N)
   for i in 1:N
      coef_PAW[:,i] = paw.coef_tilde_R(rc,i+l,l,Z)
   end
   coef_PP = paw.coef_TM(rc, l+1, l, Z, 1e-8)[1]
   B = paw.GS_VdB(N, l, Z, rc, coef_PAW, coef_PP)
   coef_proj =  paw.coef_rad_proj(B, l, Z, rc, coef_PAW, coef_PP, N)
   plot(0:0.01:rc, [polyval(coef_proj[n-l],r) for r in 0:0.01:rc])
end

function proj_custom_test(rc,n,l,Z,N)
   coef_PAW = zeros(5,N)
   for i in 1:N
      coef_PAW[:,i] = paw.coef_tilde_R(rc,i+l,l,Z)
   end
   coef_PP = paw.coef_TM(rc, l+1, l, Z, 1e-8)[1]
   B = paw.GS_custom(N, l, Z, rc, coef_PAW, coef_PP)
   plot(0:0.01:rc, [paw.rad_proj_custom(r, B, n-l, l, Z, rc, coef_PAW, coef_PP, N) for r in 0:0.01:rc])
end

function proj_custom_test_poly(rc,n,l,Z,N)
   coef_PAW = zeros(5,N)
   for i in 1:N
      coef_PAW[:,i] = paw.coef_tilde_R(rc,i+l,l,Z)
   end
   coef_PP = paw.coef_TM(rc, l+1, l, Z, 1e-8)[1]
   B = paw.GS_custom(N, l, Z, rc, coef_PAW, coef_PP)
   coef_proj =  paw.coef_proj_custom(B, l, Z, rc, coef_PAW, coef_PP, N)
   plot(0:0.01:rc, [polyval(coef_proj[n-l],r) for r in 0:0.01:rc])
end

function ortho_test_rad(rc,Npaw,Z)
   Npawtot = paw.Npawtot(Npaw)
   D = zeros(Float64,(Npawtot,Npawtot))
   coefpaw = paw.pawcoef(Z,rc,Npaw)
   function f(r,n1,l1,m1,n2,l2,m2)
      if r < rc
         return polyval(coefpaw.proj[n1-l1,l1+1],r) * paw.tilde_R_nl(r,rc,n2,l2,Z,coefpaw.tdR[:,n2-l2,l2+1])
      else
         return 0.
      end
   end
   for i1 in 1:Npawtot
      for i2 in 1:Npawtot
         l1,n1,m1 = int_to_nl(i1,Npaw)
         l2,n2,m2 = int_to_nl(i2,Npaw)
         real_f(r) = f(r,n1+l1-1,l1-1,m1,n2+l2-1,l2-1,m2)
         D[i1,i2] = QuadGK.quadgk(real_f, 0.,rc, abstol=1e-9, reltol=1e-9)[1]
      end
   end
   return D
end

function ortho_custom_test_rad(rc,Npaw,Z)
   Npawtot = paw.Npawtot(Npaw)
   D = zeros(Float64,(Npawtot,Npawtot))
   coefpaw = paw.pawcoef(Z,rc,Npaw,GS = paw.GS_custom, proj_gen = paw.coef_proj_custom)
   function f(r,n1,l1,m1,n2,l2,m2)
      if r < rc
         return polyval(coefpaw.proj[n1-l1,l1+1],r) * paw.tilde_R_nl(r,rc,n2,l2,Z,coefpaw.tdR[:,n2-l2,l2+1])
      else
         return 0.
      end
   end
   for i1 in 1:Npawtot
      for i2 in 1:Npawtot
         l1,n1,m1 = int_to_nl(i1,Npaw)
         l2,n2,m2 = int_to_nl(i2,Npaw)
         real_f(r) = f(r,n1+l1-1,l1-1,m1,n2+l2-1,l2-1,m2)
         D[i1,i2] = QuadGK.quadgk(real_f, 0.,rc, abstol=1e-9, reltol=1e-9)[1]
      end
   end
   return D
end

function ortho_test(rc,Npaw,Z)
   Npawtot = paw.Npawtot(Npaw)
   D = zeros(Complex128,(Npawtot,Npawtot))
   coefpaw = paw.pawcoef(Z,rc,Npaw)
   function f(X,n1,l1,m1,n2,l2,m2)
      if norm(X) < rc
         return conj(paw.Y_lm(X...,l1,m1))*polyval(coefpaw.proj[n1-l1,l1+1],norm(X)) * paw.Y_lm(X...,l2,m2)*paw.tilde_R_nl(norm(X),rc,n2,l2,Z,coefpaw.tdR[:,n2-l2,l2+1])/norm(X)^2
      else
         return complex(0.)
      end
   end
   for i1 in 1:Npawtot
      for i2 in 1:Npawtot
         l1,n1,m1 = int_to_nl(i1,Npaw)
         l2,n2,m2 = int_to_nl(i2,Npaw)
         real_f(X) = real(f(X,n1+l1-1,l1-1,m1,n2+l2-1,l2-1,m2))
         imag_f(X) = imag(f(X,n1+l1-1,l1-1,m1,n2+l2-1,l2-1,m2))
         D[i1,i2] = int3D(real_f,rc) + im*int3D(imag_f,rc)
      end
   end
   return D
end

function ortho_test2(rc,Npaw,Z) #other implementation of the projectors \tilde p
   Npawtot = paw.Npawtot(Npaw)
   D = zeros(Complex128,(Npawtot,Npawtot))
   coefpaw = paw.pawcoef(Z,rc,Npaw)
   function f(X,n1,l1,m1,n2,l2,m2,B,coef_PP,N)
      if norm(X) < rc
         return conj(paw.Y_lm(X...,l1,m1))*paw.rad_proj(norm(X), B, n1, l1, rc, coefpaw.tdR[:,:,l1+1], coef_PP, N)* paw.Y_lm(X...,l2,m2)*paw.tilde_R_nl(norm(X),rc,n2,l2,Z,coefpaw.tdR[:,n2-l2,l2+1])/norm(X)^2
      else
         return complex(0.)
      end
   end
   for i1 in 1:Npawtot
      for i2 in 1:Npawtot
         l1,n1,m1 = int_to_nl(i1,Npaw)
         l2,n2,m2 = int_to_nl(i2,Npaw)
         coef_PP = paw.coef_TM(rc, n1+l1-1, l1-1, 1e-8)[1]
         B = paw.GS_VdB(Npaw[l1], l1-1, rc, coefpaw.tdR[:,:,l1], coef_PP)
         N = Npaw[l1]
         real_f(X) = real(f(X,n1+l1-1,l1-1,m1,n2+l2-1,l2-1,m2,B,coef_PP,N))
#         imag_f(X) = imag(f(X,n1+l1-1,l1-1,m1,n2+l2-1,l2-1,m2))
         D[i1,i2] = int3D(real_f,rc)
      end
   end
   return D
end

function pol_proj(rc,n,l)
   L = 0.01:0.01:rc
   coef_PAW = []
   coef_PP = []
   for i in 1:2
      push!(coef_PAW, paw.coef_tilde_R(rc, i, 0))
      push!(coef_PP, paw.coef_TM(rc, i, 0, 1e-8)[1])
   end
   B = paw.GS_VdB(2, 0, rc, coef_PAW, coef_PP)
   isapprox([paw.rad_proj(r, B, n, l, rc, coef_PAW, coef_PP) for r in L], [polyval(paw.coef_rad_proj(B, n, l, rc, coef_PAW, coef_PP)[n], r) for r in L])
end

end

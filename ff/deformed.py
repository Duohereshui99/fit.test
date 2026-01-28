import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, nsolve
from sympy import cosh, sin , sinh, cos,exp
from scipy.special import factorial
from scipy.special import sph_harm_y    #sph_harm_y(l,m,theta,phi)
from scipy.optimize import fsolve,brentq
from scipy.integrate import quad,quad_vec,simpson
import numba,time
from decimal import Decimal, getcontext
from scipy.interpolate import interp1d
getcontext().prec = 50  #decimal precision


#System
e2=1.43997 ; hbarc=197.3269718 ; amu=931.49432
zp=69 ; Ap=145 ; mass_excess_p=-27.58 ; mp=Ap*amu+mass_excess_p
z1=68 ; A1=144 ; mass_excess_1=-36.61 ; m1=A1*amu+mass_excess_1    #1-> daughter
z2=1  ; A2=1   ; mass_excess_2=7.288971064 ; m2=A2*amu+mass_excess_2
Q=mp-m1-m2
mu=m1*m2/(m1+m2)
P0=0.580
#Potential

#Deformation
beta2 = 0.231  # quadrupole
beta4 = -0.068  # hexadecapole 

#Polarizaton
beta2tilde = 0  
beta4tilde = 0

#spin-orbit
Vso0=6.2 ; rso=1.01*(A1)**(1/3) ; aso=0.75 ; lambda_pi=np.sqrt(2)
L=5; S=0.5; J=5.5

#Vn
a0=0.75 ; r0=1.27*A1**(1/3)-0.1 ; V0=55

#Vc
Rc0 = 1.21 * A1**(1/3)          

#mesh
r = np.linspace(0.03, 30.0, 1000) # avoid too small r


Nlm=lambda l,m: np.sqrt((2*l+1)/(4*np.pi)*factorial(l-m)/factorial(l+m))
#derivative of Ylm on theta, theta exists on the denominator, so use epsilon to avoid division by zero

def dYlm(l,m,theta,phi):
    if np.sin(theta) <1e-10:
        return 0
    return (1/(2*l+1)/(np.sin(theta)))*(l*(l-m+1)*Nlm(l,m)/Nlm(l+1,m)*sph_harm_y(l+1,m,theta,phi)-(l+1)*(l+m)*Nlm(l,m)/Nlm(l-1,m)*sph_harm_y(l-1,m,theta,phi))
#deformed potential,with deformed paras beta, polarization params beta_tilde
#Avoid using zero theta, if needed, set theta=1e-10
def Vn(V0,r0,a0,beta2,beta4,beta2_tilde,beta4_tilde,r,theta):
    Y20=sph_harm_y(2,0,theta,0)  #sph_harm_y(l,m,theta,phi)
    Y40=sph_harm_y(4,0,theta,0)
    r0_theta=r0*(1+beta2*Y20+beta4*Y40)         #deformation

    # delta_theta=1e-6
    # dY20=(sph_harm(0, 2, 0, theta+delta_theta)-sph_harm(0, 2, 0, theta))/(delta_theta)
    # dY40=(sph_harm(0, 4, 0, theta+delta_theta)-sph_harm(0, 4, 0, theta))/(delta_theta)
    dY20=dYlm(2,0,theta,0)
    dY40=dYlm(4,0,theta,0)
    dr0_theta=r0*(beta2*dY20+beta4*dY40)       

# polarization
    a0_theta=a0*np.sqrt(1+(1/r0_theta*dr0_theta)**2)*(1+beta2_tilde*Y20+beta4_tilde*Y40) 

    return -V0/(1+np.exp((r-r0_theta)/a0_theta))

#V_{so}, with lambda_pi being the wave number of pion
def Vso(Vso0,rso,aso,L,S,J,r,theta,beta2,beta4,beta2_tilde,beta4_tilde):
    lambda_pi2=2
    Y20=sph_harm_y(2,0,theta,0)  #sph_harm_y(l,m,theta,phi)
    Y40=sph_harm_y(4,0,theta,0)
    dY20=dYlm(2,0,theta,0)
    dY40=dYlm(4,0,theta,0)
    #deformation
    rso=rso*(1+beta2*Y20+beta4*Y40) 

    #polarization
    drso=rso*(beta2*dY20+beta4*dY40)       
    aso=aso*np.sqrt(1+(1/rso*drso)**2)*(1+beta2_tilde*Y20+beta4_tilde*Y40) 
    return Vso0*lambda_pi2/aso*(-np.exp((r-rso)/aso)/(r*(1+np.exp((r-rso)/aso))**2))*((J*(J+1)-L*(L+1)-S*(S+1))/2)*2

def Vc_vectorized(z1, z2, r, theta, Rc0, beta2, beta4):

    r = np.asarray(r)               
    
    Y00 = sph_harm_y(0, 0, theta, 0).real
    Y20 = sph_harm_y(2, 0, theta, 0).real
    Y40 = sph_harm_y(4, 0, theta, 0).real

# theta grid
    nt = 300  
    t_grid = np.linspace(0, np.pi / 2, nt)
    sin_t = np.sin(t_grid)

    # Rc(t)
    Rc_t = Rc0 * (1 + beta2 * sph_harm_y(2, 0, t_grid, 0).real + beta4 * sph_harm_y(4, 0, t_grid, 0).real)  # (nt,)
    #K_lambda(theta,r) (nt,nr)
    def compute_res(lam):
        Yl0_t = sph_harm_y(lam, 0, t_grid, 0).real  # (nt,)
        K = np.zeros((nt, len(r)))  # (nt, nr)
        
        for i in range(nt):
            Rc = Rc_t[i]
            inside = r <= Rc
            if lam == 2:
                # 原版 log 项
                K[i, inside]  = r[inside]**2 / 5 + r[inside]**2 * np.log(Rc / r[inside])
                K[i, ~inside] = Rc**(lam + 3) / ((lam + 3) * r[~inside]**(lam + 1))
            else:
                K[i, inside]  = (2 * lam + 1) * r[inside]**2 / ((lam + 3) * (lam - 2)) - r[inside]**lam / ((lam - 2) * Rc**(lam - 2))
                K[i, ~inside] = Rc**(lam + 3) / ((lam + 3) * r[~inside]**(lam + 1))
        
        integrand = Yl0_t[:, np.newaxis] * K * sin_t[:, np.newaxis]  # (nt, nr)
        res = 2 * simpson(integrand, x=t_grid, axis=0)  # (nr,)
        return res

    res0 = compute_res(0)
    res2 = compute_res(2)
    res4 = compute_res(4)


    V_0 = 3 * z1 * z2 * e2 / Rc0**3 * 2 * np.pi * Y00 * res0
    V_2 = 3 * z1 * z2 * e2 / Rc0**3 * 2 * np.pi * Y20 / 5 * res2
    V_4 = 3 * z1 * z2 * e2 / Rc0**3 * 2 * np.pi * Y40 / 9 * res4
    
    Vc_tot = V_0 + V_2 + V_4

    return Vc_tot, V_0, V_2, V_4



def compute_gamma_for_theta(theta, m1, m2, z1, z2, Rc0, V0, a0, r0, P0, beta2, beta4, beta2tilde, beta4tilde):

#mesh
    r_values = np.linspace(0.1, 100.0, 300)   # for searching roots
    r_grid   = np.linspace(0.05, 100.0, 1000) # for interpolation

#pre calculate Vn, Vso, Vc
    Vn_grid   = Vn(V0, r0, a0, beta2, beta4, beta2tilde, beta4tilde, r_grid, theta)
    Vso_grid  = Vso(Vso0, rso, aso, L, S, J, r_grid, theta, beta2, beta4, beta2tilde, beta4tilde)
    Vc_grid, _, _, _ = Vc_vectorized(z1, z2, r_grid, theta, Rc0, beta2, beta4)
    
#interpolate Vn, Vso, Vc
    interp_Vn  = interp1d(r_grid, Vn_grid,   kind='cubic', fill_value="extrapolate", bounds_error=False)
    interp_Vso = interp1d(r_grid, Vso_grid,  kind='cubic', fill_value="extrapolate", bounds_error=False)
    interp_Vc  = interp1d(r_grid, Vc_grid,   kind='cubic', fill_value="extrapolate", bounds_error=False)

    def f(r):
        V_cent = hbarc**2 * (L + 0.5)**2 / (2 * mu * r**2)
        return interp_Vn(r) + interp_Vso(r) + interp_Vc(r) + V_cent - Q

    def k(r):
        V_cent = hbarc**2 * (L + 0.5)**2 / (2 * mu * r**2)
        Vtot = interp_Vn(r) + interp_Vso(r) + interp_Vc(r) + V_cent
        arg = (2 * mu / hbarc**2) * np.abs(Vtot - Q)
        return np.sqrt(np.maximum(arg, 0.0))

    def integrand(r):
        return 1 / (2 * k(r))

# Find roots
    f_values = f(r_values)
    roots = []
    for i in range(len(r_values)-1):
        if np.sign(f_values[i]) != np.sign(f_values[i+1]):
            try:
                root = brentq(f, r_values[i], r_values[i+1], xtol=1e-6)
                if root > 0 and not any(np.isclose(root, r, atol=1e-5) for r in roots):
                    roots.append(root)
            except ValueError:
                pass
    roots.sort()

    if len(roots) < 3:
        return np.nan  

# Integrate
    integral_result, _ = quad(integrand, roots[0], roots[1], limit=50)
    F = 1 / integral_result

    action, _ = quad(k, roots[1], roots[2], limit=50)
    Gamma = P0 * F * hbarc**2 / (4 * mu) * np.exp(-2 * action)

# Gamma(theta)
    return Gamma

def Model(m1, m2, z1, z2, Rc0, V0, a0, r0, P0, beta2, beta4, beta2tilde, beta4tilde, n_theta=30):

    theta_grid = np.linspace(0, np.pi/2, n_theta)
    sin_theta = np.sin(theta_grid)

    gamma_list = []
    for idx, theta in enumerate(theta_grid, 1):

        gamma = compute_gamma_for_theta(
            theta, m1, m2, z1, z2, Rc0, V0, a0, r0, P0, beta2, beta4, beta2tilde, beta4tilde
        )
        gamma_list.append(gamma)

    Gamma_array = np.array(gamma_list)
    integrand = Gamma_array * sin_theta
    Gamma_total = simpson(integrand, x=theta_grid)


# Gamma(theta) and Gamma
    return theta_grid, Gamma_array, Gamma_total



theta_arr, Gamma_arr, Gamma_ave = Model(m1, m2, z1, z2, Rc0, V0, a0, r0, P0, beta2, beta4, beta2tilde, beta4tilde)

print(f"Total width: {Gamma_ave:.3e} MeV")
print(hbarc*np.log(2)/Gamma_ave*1/3*1e-23)
plt.plot(theta_arr, Gamma_arr)
plt.xlabel("θ (rad)")
plt.ylabel("Γ (MeV)")
plt.title("Gamma vs θ")
plt.show()
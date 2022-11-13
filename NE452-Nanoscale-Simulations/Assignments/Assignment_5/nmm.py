import numpy as np
from sys import argv, exit
import matplotlib.pyplot as plt

def kinetic(size,mass,dx):
    T=np.zeros((size,size),float)
    h_bar=1.
    for i in range(size):
        for ip in range(size):
            T1 = h_bar*h_bar / (2.0*mass*dx*dx) * np.power(-1.,i-ip);
            if i==ip:
                T[i,ip] = T1 * (np.pi*np.pi/3.0);
            else:
                T[i,ip] = T1 * 2.0/((i-ip)*(i-ip))
    return T

mass=1
hbar=1.
omega=1.

# beta=1/kBT value
beta = float(argv[1])
P = int(argv[2])
tau=beta/P
xmax = float(argv[3])

size = int(argv[4])
dx=2.*xmax/(size-1.)
grid=np.zeros(size,float)
for i in range(size):
     grid[i]=-xmax+i*dx

pot_type = argv[5]

yliml=-1.
ylimh=20.
if pot_type == 'HO':
    def V(x):
        return (1./2.)*(mass*omega**2)*x**2
    print('------------------------------------------------------------------------------------------')
    print('For the harmonic oscillator, we know the partition function and free energy exactly')
    print('Z(analytical;harmonic,beta='+str(beta)+')= ',1./(2.*np.sinh(beta*hbar*omega/2.)))
    print('A(analytical;harmonic,beta='+str(beta)+')= ',-np.log(1./(2.*np.sinh(beta*hbar*omega/2.)))/beta)
    print('------------------------------------------------------------------------------------------')
elif pot_type == 'DW':
    a = -.5
    b = .1
    def V(x):
        return a*x**2+b*x**4
else:
    print('Incorrect potential specified')
    exit()

#Build the free particle density matrix
rho_free=np.zeros((size,size),float)
for i1,x1 in enumerate(grid):
    for i2,x2 in enumerate(grid):
        rho_free[i1,i2]=np.sqrt(mass/(2.*np.pi*tau*hbar*hbar))*(np.exp(-(mass/(2.*hbar*hbar*tau))*(x1-x2)*(x1-x2)))

#Build the potential density matrix
rho_potential=np.zeros((size),float)
potential=np.zeros((size),float)
for i1,x1 in enumerate(grid):
    potential[i1]=V(x1)
    rho_potential[i1]=np.exp(-(tau/2.)*potential[i1])

#Output the potential to a file
potential_out=open('V_'+pot_type+'_'+str(beta)+'_'+str(tau),'w')
for i1,x1 in enumerate(grid):
	potential_out.write(str(x1)+' '+str(potential[i1])+'\n')
potential_out.close()

# construct the high temperature density matrix
rho_tau=np.zeros((size,size),float)
for i1 in range(size):
    for i2 in range(size):
        rho_tau[i1,i2]=rho_potential[i1]*rho_free[i1,i2]*rho_potential[i2]

# form the density matrix via matrix multiplication
#set initial value of rho
rho_beta=rho_tau.copy()

for k in range(P-1):
    rho_beta=dx*np.dot(rho_beta,rho_tau)

# calculate partition function from the trace of rho
Z=0.
Z_tau=0.
V_estim=0.
for i in range(size):
    Z_tau+=rho_tau[i,i]*dx
    Z+=rho_beta[i,i]*dx
    V_estim+=rho_beta[i,i]*dx*potential[i]

print('------------------------------------------------------------------------------------------')
print('Path integral matrix multiplication')
print('Z(beta=',beta,',tau=',tau,')= ',Z)
print('A(beta=',beta,',tau=',tau,')= ',-np.log(Z)/beta)
print('------------------------------------------------------------------------------------------')

rho_x = []
rho_y = []
rho_beta_out=open('rho_'+pot_type+'_'+str(beta)+'_'+str(tau),'w')
for i1,x1 in enumerate(grid):
    rho_x.append(grid[i1])
    rho_y.append(rho_beta[i1,i1]/Z)
    #rho_beta_out.write(str(x1)+' '+str(rho_beta[i1,i1]/Z/x1/x1)+'\n')
    rho_beta_out.write(str(x1)+' '+str(rho_beta[i1,i1]/Z)+'\n')
rho_beta_out.close()

#plt.plot(rho_x,rho_y)
#plt.savefig('test.pdf')


# compare Z Trotter, Analytical, truncated sum over states
# Hamiltonian matrix
H=kinetic(size,mass,dx)
for i in range(size):
	    H[i,i]+=potential[i]
eig_vals,eig_vecs=np.linalg.eig(H)

ix = np.argsort(eig_vals)
eig_vals_sorted = eig_vals[ix]
eig_vecs_sorted = eig_vecs[:,ix]

# sum over states
Z_sos=0.
for e in eig_vals_sorted:
	Z_sos+=np.exp(-beta*e)

print('------------------------------------------------------------------------------------------')
print('Sum over states solution')
print('Z_sos= ',Z_sos)
print('A_sos(beta=',beta,')= ',-np.log(Z_sos)/beta)
print('------------------------------------------------------------------------------------------')






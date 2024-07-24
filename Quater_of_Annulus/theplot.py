from bspline   import elements_spans  
from bspline   import make_knots,basis_funs,find_span     
from bspline   import quadrature_grid ,breakpoints
from bspline   import basis_ders_on_quad_grid,basis_funs_all_ders 
from Gauss_Legendre import Gauss_Legendre, quadrature_grid
from stdio     import Mass_Matrix, Stiffness_Matrix, assemble_rhs_with_Non_homogenuous_DBC
from stdio     import B_Spline_Least_Square, assemble_stiffness_2D, L2_projection, L2_projectionVec
from equipment import L2_norm_2D, H1_norm_2D, plot_field_2D
from scipy.sparse.linalg import cg
from scipy.linalg import norm, inv, solve, det,inv
from numpy import zeros, ones, linspace,double,float64, cos,array, dot, zeros_like, asarray,floor,arange,append,random,sqrt, int32, meshgrid,sin
import  matplotlib.pyplot as plt
from scipy.sparse        import csr_matrix
from scipy.sparse        import csc_matrix, linalg as sla
from scipy.sparse.linalg import gmres
from numpy               import zeros, linalg, asarray
from numpy               import cos, pi

#from matplotlib.pyplot import plot, show
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pyccel.decorators import types
                   
from scipy.sparse import kron as spkron





@types('double[: , : , : , :]', 'int', 'int')
def tensor_to_matrix(Mat, nh, nm) :
    sti = zeros((nh * nm, nh * nm), dtype=double)
    for i in range(nh):
        for j in range(nm):
            for k in range(nh):
                for l in range(nm):
                    i_1 = k + i * nm
                    i_2 = l + j * nh
                    sti[i_1, i_2] = Mat[i, j, k, l]                    
    return sti


#Test 1 on circule  homogenous Boundary Conditions




# Test 2

Uex = lambda x, y : 1.-x**2-y**2
dUx = lambda x, y : -2*x 
dUy = lambda x, y : -2*y
f   = lambda x, y : 0.*x+4.
                   
#Test 2 



#Test 2 Non homogenous Boundary Conditions

F1= lambda x1, x2 : (x1+1.0)*cos(0.5*pi*x2)
F2= lambda x1, x2 : (x1+1.0)*sin(0.5*pi*x2)

Uex = lambda x, y : 2.*x*cos(pi*y) 
dUx = lambda x, y : 2.*cos(pi*y)  
dUy = lambda x, y : -2*pi*x*sin(pi*y)
f   = lambda x, y : 2.*x*pi**2*cos(pi*y)

# Test 3 Non homogenous Boundary Conditions
'''

kappa = 2

Uex   = lambda x,y :  cos(pi*kappa*x)- sin(2*kappa*pi*y)
dUx   = lambda x,y : -pi*kappa*sin(pi*kappa*x)
dUy   = lambda x,y : - 2*kappa*pi*cos(2*kappa*pi*y)
f     = lambda x,y : (pi*kappa)**2*cos(pi*kappa*x)- (2*kappa*pi)**2*sin(2*kappa*pi*y)
'''
p1, p2                   = (2 , 2)
ne1, ne2                = (32 , 32)
a, b, c, d = 0., 1., 0., 1.
grid1, grid2              = linspace(a, b, ne1+1), linspace(c, d, ne2+1)            
knots1, knots2           = make_knots(grid1, p1, False), make_knots(grid2, p2, False)
spans1, spans2           = elements_spans(knots1, p1), elements_spans(knots2, p2)
nelements1, nelements2  = len(grid1) - 1 , len(grid2) - 1
nbasis1, nbasis2         = len(knots1) - p1 - 1,  len(knots2) - p2 -1
nders                    = 1
U1 , W1                 = Gauss_Legendre(p1)
U2 , W2                 = Gauss_Legendre(p2)
points1, weights1         = quadrature_grid(grid1,U1,W1)
points2, weights2         = quadrature_grid(grid2,U2,W2)
basis1, basis2             = basis_ders_on_quad_grid(knots1, p1, points1, nders, normalize=False),basis_ders_on_quad_grid(knots2, p2, points2, nders, normalize=False)
stiffness                   = zeros((nbasis1, nbasis2, nbasis1, nbasis2))
stiffness     = assemble_stiffness_2D(nelements1, nelements2, p1, p2, spans1, spans2, basis1, basis2, weights1, weights2, points1, points2, stiffness)
stiffness     = tensor_to_matrix(stiffness[1:-1,1:-1,1:-1,1:-1], nbasis1-2, nbasis2-2)


rhs1          = zeros((nbasis1 , nbasis2))

# here to applay non homogenous use this 

X      = linspace(knots1[0], knots1[-p1], nbasis1)
gx0    = Uex(F1(0,X),F2(0,X))
gx0_h         = L2_projectionVec(knots2, p2, X, gx0)
gx1    = Uex(F1(1,X),F2(1,X))
gx1_h         = L2_projectionVec(knots2, p2, X, gx1)
gy0    = Uex(F1(X,0),F2(X,0))
gy0_h         = L2_projectionVec(knots1, p1, X, gy0)
gy1    = Uex(F1(X,1),F2(X,1))
gy1_h         = L2_projectionVec(knots1, p1, X, gy1)

g_bou         = zeros((nbasis1, nbasis2), dtype = double)
g_bou[0,:]    = gx0_h
g_bou[-1,:]   = gx1_h
g_bou[:,0]    = gy0_h
g_bou[:,-1]   = gy1_h

rhs1          = assemble_rhs_with_Non_homogenuous_DBC(f, g_bou, nelements1, nelements2, p1, p2, spans1, spans2, basis1, basis2, weights1, weights2, points1, points2, rhs1)
rhs1          = rhs1[1:-1, 1:-1]
rhs1          = rhs1.reshape((nbasis1-2)*(nbasis2-2))
lu            = sla.splu(csc_matrix(stiffness))
Uapp          = lu.solve(rhs1) 
Uh            = zeros((nbasis1,nbasis2))
Uh[1:-1,1:-1] = Uapp.reshape((nbasis1-2),(nbasis2-2))
Uh            = Uh + g_bou
#####################################For the plot approximation solution#################################################
@types('double[:]', 'double[:]', 'int', 'int', 'double[:]', 'int', 'int')
def plot_field_2D(knots1, knots2, degree1, degree2, u, nx = 101, ny=101):
    xmin, xmax, ymin ,ymax = knots1[degree1], knots1[-degree1], knots2[degree2], knots2[-degree2]
    
    xs = linspace(xmin, xmax, nx)
    ys = linspace(ymin, ymax, ny)
    
    P = zeros((nx,ny,1))
    nu1,nu2 = u.shape
    Q = zeros((nu1,nu2))
    Q[:,:]= u[:,:]
    for i ,xi in enumerate(xs):
        for j,yj in enumerate(xs):
            i_span = find_span(knots1, degree1, xi)
            j_span = find_span(knots2, degree2, yj)
            valuesx = basis_funs(knots1, degree1, xi, i_span)
            valuesy = basis_funs(knots2, degree2, yj, j_span)
            C = zeros(P.shape[-1])
            for i_1 in range(degree1+1):
                for j_1 in range(degree2+1):
                    C[:]+=valuesx[i_1]*valuesy[j_1]*Q[i_span-degree1+i_1, j_span-degree2+j_1]
            P[i , j, :] = C[:]
    X,Y = meshgrid(xs,ys)
    ax = plt.axes(projection = '3d')
    Xc        =F1(X, Y)
    Yc        = F2(X, Y)
    #plt.contourf(Xc,Yc, P[:,:,0].T)
    #ax.plot_surface(Xc,Yc, P[:,:,0].T, cmap ='viridis', edgecolor ='yellow')

    ax.plot_surface(Xc, Yc, P[:,:,0].T, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # ax.set_title('Approximate solution')
    ax.set_xlabel('Xc',  fontweight ='bold')
    ax.set_ylabel('Yc',  fontweight ='bold')
    plt.savefig('Poisson3D_Appro.png')
    # Add a color bar which maps values to colors.
    plt.show()
    
plot_field_2D(knots1,knots2, p1, p2, Uh)
l2 = L2_norm_2D( nelements1, nelements2, p1, p2, spans1, spans2, basis1, basis2, weights1, weights2, points1, points2, Uh, Uex)
print('#############################################',' p1=p1 =',p1,'and  ne1 = ne2 = ', ne1,'##############################################')
print('\n')
print('The L2 norm for p1=p1 =',p1,'and  ne1 = ne2 = ', ne1, ' is equal to ', l2)
h1 = H1_norm_2D( nelements1, nelements2, p1, p2, spans1, spans2, basis1, basis2, weights1, weights2, points1, points2, Uh, dUx, dUy, Uex)
print('\n')
print('The H1 norm for p1=p1 =',p1,'and  ne1 = ne2 = ', ne1, ' is equal to ', h1)
#####################################For the plot Exact solution#################################################

nbpts = 250
xs = ys = linspace(0., 1., nbpts)
X, Y = meshgrid(xs, ys)
Xc        =F1(X, Y)
Yc        = F2(X, Y)
sol = zeros((nbpts,nbpts))
for i in range(nbpts) :
     for j in range(nbpts) :
         sol[i, j] = Uex(Xc[i,j], Yc[i,j])
#sol =  Uex(Xc, Yc)
ax = plt.axes(projection = '3d')
    
#ax.plot_surface(Xc,Yc, sol, cmap ='viridis', edgecolor ='yellow')
ax.plot_surface(Xc, Yc, sol, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.0, 1.0)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_xlabel('F1',  fontweight ='bold')
ax.set_ylabel('F2',  fontweight ='bold')
plt.show()
plt.savefig('Poisson3D.png')
# plt.contourf(X,Y, sol)



















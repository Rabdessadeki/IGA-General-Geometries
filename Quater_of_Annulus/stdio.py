#Functions needed from numpy and scipy
from numpy import kron, zeros, linspace, empty, double, sqrt, cos, sin, pi
from scipy.sparse  import csc_matrix, linalg as sla
from pyccel.decorators import types         

__Global__= ['Mass_Matrix',
             'Stiffness_Matrix',
             'assemble_rhs',
             'assemble_stiffness_2D',
             'assemble_rhs_with_Non_homogenuous_DBC',
             'B_Spline_Least_Square'
]

#########################Assemble stiffness and mass matrix 1D#######################################
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def Mass_Matrix(nelements, degree, spans, basis, weights, points, matrix):
    k1          = weights.shape[1]
    ne          = nelements
    p           = degree      
    spans       = spans
    weights     = weights
    basis       = basis
    points      = points
    for i_e in range(ne):
        i_span = spans[i_e]
        for i_1 in range(p+1):
            i  = i_span - p + i_1
            for j_1 in range( p + 1):
                j  = i_span - p + j_1
                v0         = 0.0
                for i_k1 in range(k1):
                    b_xi   = basis[i_e, i_1, 0, i_k1] * basis[i_e, j_1, 0, i_k1]
                    v0    += (b_xi ) * weights[i_e, i_k1]
                matrix[i , j] += v0
    return matrix
    
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def Stiffness_Matrix(nelements, degree, spans, basis, weights, points, matrix):
    k1          = weights.shape[1]
    ne          = nelements
    p           = degree      
    spans       = spans
    weights     = weights
    basis       = basis
    points      = points
    for i_e in range(ne):
        i_span = spans[i_e]
        for i_1 in range( 0, p + 1 ):
            i  = i_span - p + i_1
            for j_1 in range( 0, p + 1):
                j   = i_span - p + j_1
                v0  = 0.0
                for i_k1 in range( 0,  k1 ):
                    b_xi  = basis[i_e, i_1, 1, i_k1] *  basis[i_e, j_1, 1, i_k1]                    
                    v0  += ( b_xi ) * weights[i_e, i_k1]
                matrix[i , j] += v0
    return matrix

#########################Assemble rhs 1D non_homogenenous boundary #######################################
@types('double', 'int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:]', 'double[:]')
def assemble_rhs_1d_with_Non_homg_Dirichlet(f, nelements, degree, spans, basis, weights, points, vectu, rhs):

    ne1       = nelements
    p1        = degree
    spans_1   = spans
    basis_1   = basis
    weights_1 = weights
    points_1  = points
    k1        = weights.shape[1]

    coef_u = zeros(k1)
    for ie1 in range(0, ne1):
        i_span_1  = spans_1[ie1]
        coef_u[:] = vectu[i_span_1-p1:i_span_1+1]
        for g1 in range(0,k1):
            sx=0.
            for il_1 in range(0,p1+1):
                   bi_x  = basis_1[ie1, il_1, 1, g1]
                   coef  = coef_u[il_1]
                   sx   += coef*bi_x
            values[g1]   = sx
        for il_1 in range(0, p1+1):
            i1 = i_span_1 - p1 + il_1
            v = 0.0
            for g1 in range(0, k1):
                bi_0   = basis_1[ie1, il_1, 0, g1]
                bi_x   = basis_1[ie1, il_1, 1, g1]  

                x1     = points_1[ie1, g1]
                wvol   = weights_1[ie1, g1]
                sx     = values[g1]

            v       += bi_0 * f(x1) * wvol - bi_x*sx*wvol
            rhs[i1] += v
            
    return rhs
    
################################## 2D assembling: stiffness######################################## 

@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:,:,:]')
def assemble_stiffness_2D(ne1, ne2, p1, p2,  spans1, spans2, basis1, basis2, weights1, weights2, points1, points2, matrix):
    k1                             = weights1.shape[1]
    k2                             = weights2.shape[1]
    for i_e1 in range(ne1):
        i_span1 = spans1[i_e1]
        
        for i_e2 in range(ne2):
            i_span2 = spans2[i_e2]
            
            for i_1 in range(p1+1):
                i = i_span1 - p1 + i_1
                
                for j_1 in range(p2+1):
                    j = i_span2 - p2 +  j_1
                    
                    for i_2 in range(p1+1):
                        k = i_span1 - p1 + i_2
                        
                        for j_2 in range(p2+1):                            
                            l = i_span2 - p2 + j_2
                            
                            punch = 0.0
                            for i_k1 in range(k1):
                                for i_k2 in range(k2):
                                    x1 = points1[i_e1, i_k1]
                                    x2 = points1[i_e2, i_k2]
                                    x =  (x1+1.0)*cos(0.5*pi*x2)
                                    y=   (x1+1.0)*sin(0.5*pi*x2)
                                    
                                    
                                    F1x      = cos(0.5*pi*x2)
                                    F1y      = -0.5*pi*(x1+1)*sin(0.5*pi*x2)
                                    F2y      = 0.5*pi*(x1+1)*cos(0.5*pi*x2)
                                    F2x      = sin(0.5*pi*x2)
                                    
                                    #det_Hess = abs(F1x*F2y-F1y*F2x)
                                    det_Hess = 0.5*pi*(x1+1)
                                      
                                    bdx_= basis1[i_e1, i_1, 1, i_k1] * basis2[i_e2, j_1, 0, i_k2]
                                    by_   = basis1[i_e1, i_1, 0, i_k1] * basis2[i_e2, j_1, 1, i_k2]
                                    bx_   = basis1[i_e1, i_2, 1, i_k1] * basis2[i_e2, j_2, 0, i_k2]
                                    bdy_  = basis1[i_e1, i_2, 0, i_k1] * basis2[i_e2, j_2, 1, i_k2]                                    
                                    
                                    b1_x  = F2y*bdx_ - F2x*by_
                                    b1_y  = -F1y*bdx_ +F1x*by_                                    
                                    b2_x  = F2y*bx_ - F2x*bdy_                                   
                                    b2_y  = -F1y*bx_ +F1x*bdy_
                                                                        
                                    weitr = weights1[i_e1, i_k1] * weights2[i_e2, i_k2]
                                    punch +=( (b1_x * b2_x ) +  ( b1_y* b2_y ) ) * weitr/det_Hess
                            # matrix[i + k*spline_number1, l + j*spline_number2]+= punch
                            matrix[i, k, j, l]+= punch
    return matrix
    
################################## 2D assembling: rhs with homogenou########################################
@types('double', 'double[:,:]','int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')

def assemble_rhs_with_Non_homogenuous_DBC(f, g_bou, ne1, ne2, p1, p2,  spans1, spans2, basis1, basis2, weights1, weights2, points1, points2, rhs):
    
    k1                      = weights1.shape[1]
    k2                      = weights2.shape[1]
    g_1                     = zeros((k1,k2), dtype = double)
    g_2                     = zeros((k1,k2), dtype = double)
    values_boun             = zeros((p1+1,p2+1),dtype = double)
    for i_e1 in range(0, ne1):
        for i_e2 in range(0, ne2):
            i_span1 = spans1[i_e1]
            i_span2 = spans2[i_e2] 
            values_boun[: , :] = g_bou[i_span1-p1 : i_span1+1 , i_span2-p2 : i_span2 +1 ]
            
            for i_k1 in range(0, k1):
                x1  = points1[i_e1,i_k1]
                
                for i_k2 in range(0, k2):
                    x2 = points2[i_e2, i_k2]
                    
                    S1 = 0.0;S2 = 0.0
                    for i_1 in range(p1+1):
                        for j_1 in range(p2+1): 
                                               
                            i    = i_span1 - p1 + i_1
                            j    = i_span2 - p2 + j_1
                            bx_j = basis1[i_e1, i_1, 1, i_k1]*basis2[i_e2, j_1, 0, i_k2]
                            by_j = basis1[i_e1, i_1, 0, i_k1]*basis2[i_e2, j_1, 1, i_k2]
                            g_ij = values_boun[i_1, j_1]                            
                            S1  += bx_j * g_ij  #j
                            S2  += by_j * g_ij#j                            
                    g_1[i_k1 , i_k2] = S1   
                    g_2[i_k1 , i_k2] = S2 
                    
            for i_1 in range(p1+1):
                for j_1 in range(p2+1):
                    i = i_span1 - p1 + i_1
                    j = i_span2 - p2 + j_1
                    saved = 0.0
                    
                    for i_k1 in range(0, k1):
                        x1  = points1[i_e1,i_k1]
                        
                        for i_k2 in range(0, k2):
                            x2 = points2[i_e2, i_k2]  
                            
                            x =  (x1+1.0)*cos(0.5*pi*x2)
                            y =  (x1+1.0)*sin(0.5*pi*x2)

                            F1x      = cos(0.5*pi*x2)
                            F1y      = -0.5*pi*(x1+1)*sin(0.5*pi*x2)
                            F2y      = 0.5*pi*(x1+1)*cos(0.5*pi*x2)
                            F2x      = sin(0.5*pi*x2)
                            
                            #det_Hess = abs(F1x*F2y-F1y*F2x)
                            det_Hess = 0.5*pi*(x1+1)
                            
                            b_0 = basis1[i_e1, i_1, 0, i_k1]* basis2[i_e2, j_1, 0, i_k2]                                           
                            bdx_= basis1[i_e1, i_1, 1, i_k1] * basis2[i_e2, j_1, 0, i_k2]
                            by_   = basis1[i_e1, i_1, 0, i_k1] * basis2[i_e2, j_1, 1, i_k2]
                            
                            b1_x  =F2y*bdx_ - F2x*by_ 
                            b1_y  = -F1y*bdx_ +F1x*by_
                            
                            g2_x  = F2y*g_1[i_k1 , i_k2] - F2x*g_2[i_k1 , i_k2]
                            g2_y  = -F1y*g_1[i_k1 , i_k2] +F1x*g_2[i_k1 , i_k2]

                           
                            weitr = weights1[i_e1, i_k1] * weights2[i_e2, i_k2]                        
                            saved+=  b_0 *f( x , y)*det_Hess*weitr - ( g2_x* b1_x + g2_y* b1_y) * weitr/det_Hess     
                    # rhs[i+j*spline_number]+=saved
                    rhs[i , j] +=saved
    return rhs
    
############################################BSpline Least Square approximation#############################################################
@types('int', 'double' ,'double[:,:]',  'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]')

def B_Spline_Least_Square(ne, g, p, spans, basis, weights, points):
    # ... sizes
   
    k1                   = weights.shape[1]
    rhs                  = zeros( ne + p )
    matrix = zeros((ne + p, ne + p ))
    for i_e in range(ne):
        i_span = spans[i_e]
        for i_l in range( p + 1 ):
            i_1 = i_span - p + i_l
            for j_l in range( p + 1 ):
                j_1 = i_span - p + 1
                punch0     = 0.0
                for i_k in range( k1 ):
                    bx_0       = basis[i_e, i_l, 0, i_k] * basis[i_e, j_l, 0, i_k]
                    val_wei    = weights[i_e, i_k]
                    punch0     += bx_0 * val_wei
                matrix[i_1 , j_1] += punch0

            punch1 = 0.0
            for i_k in range( k1 ):
                bx_0       = basis[i_e, i_l, 0, i_k]
                x1         = points[i_e, i_k]
                val_wei    = weights[i_e, i_k]
                punch1    += bx_0 * g( x1 )* val_wei
                
            rhs[i_1] += punch1

    lu    = sla.splu(csc_matrix(matrix))
    gh    = lu.solve(rhs) 
    return gh  
########################### Least Square B-spline ####################################################################################
@types('int', 'double' ,'double[:,:]',  'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]')

def B_Spline_Least_Square(ne, g, p, spans, basis, weights, points):
    # ... sizes
    
    k1                   = weights.shape[1]
    rhs                  = zeros( ne + p )
    matrix               = zeros((ne + p, ne + p ))
    for i_e in range(ne):
        i_span = spans[i_e]
        for i_l in range( p + 1 ):
            i_1 = i_span - p + i_l
            
            for j_l in range( p + 1 ):
                j_1 = i_span - p + j_l
                punch0     = 0.0
                for i_k in range( k1 ):
                    bx_0       = basis[i_e, i_l, 0, i_k] * basis[i_e, j_l, 0, i_k]
                    val_wei    = weights[i_e, i_k]
                    punch0     += bx_0 * val_wei
                    
                matrix[i_1 , j_1] += punch0
                
            punch1 = 0.0
            for i_k in range( k1 ):
                bx_0       = basis[i_e, i_l, 0, i_k]
                x1         = points[i_e, i_k]
                val_wei    = weights[i_e, i_k]
                punch1    += bx_0 * g( x1 )* val_wei
                
            rhs[i_1] += punch1

    lu    = sla.splu(csc_matrix(matrix))
    gh    = lu.solve(rhs) 
    return gh  
########################### B-spline L2 projection ####################################################################################
@types('double[:]', 'int', 'double')
def find_span( knots, p, x ):
  
    low   = p
    high = len(knots)-1-p
    if   x <= knots[low ]: mid = low
    elif x >= knots[high]: mid =  high-1
    else:
        mid = (low+high)//2
        while x < knots[mid] or x >= knots[mid+1]:
            if x < knots[mid]:
               high = mid
            else:
               low  = mid
            mid = (low+high)//2

    return mid
@types('double[:]', 'int', 'double', 'int')
def basis_funs( knots, degree, x, span ):

    left   = empty( degree  , dtype=double )
    right  = empty( degree  , dtype=double )
    values = empty( degree+1, dtype=double )

    values[0] = 1.0
    for j in range(0,degree):
        left [j] = x - knots[span-j]
        right[j] = knots[span+1+j] - x
        saved    = 0.0
        for r in range(0,j+1):
            temp      = values[r] / (right[r] + left[j-r])
            values[r] = saved + right[r] * temp
            saved     = left[j-r] * temp
        values[j+1] = saved

    return values

@types('double[:]', 'int', 'double')
def L2_projection(knots, p, g):
    nbasis = len(knots) - p - 1
    X      = linspace(knots[0], knots[-p], nbasis)
    matrix      = zeros((nbasis, nbasis))
    G      = zeros(nbasis)
    for i ,ix in enumerate(X):
        i_span    = find_span(knots, p, ix)
        values_xi = basis_funs(knots, p,  ix, i_span)
        matrix[i , i_span- p: i_span+1 ] = values_xi[:]
        G[i]                   = g(ix)
    lu    = sla.splu(csc_matrix(matrix))
    gh    = lu.solve(G) 
    return gh 

@types('double[:]', 'int', 'double')
def L2_projectionVec(knots, p, X, G):
    nbasis = len(knots) - p - 1
    #X      = linspace(knots[0], knots[-p], nbasis)
    matrix      = zeros((nbasis, nbasis))
    for i ,ix in enumerate(X):
        i_span    = find_span(knots, p, ix)
        values_xi = basis_funs(knots, p,  ix, i_span)
        matrix[i , i_span- p: i_span+1 ] = values_xi[:]
       
        #G[i]                   = g(ix)
    lu    = sla.splu(csc_matrix(matrix))
    gh    = lu.solve(G) 
    return gh 
























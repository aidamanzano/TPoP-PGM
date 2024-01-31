import numpy as np
import mayavi.mlab as mlab


def test_surf():
    """Test surf on regularly spaced co-ordinates like MayaVi."""
    def f(x, y):
        
        return x*y
    
    x, y = np.mgrid[0:1:0.1, 0:1:0.1]
    z= x * y
    z2 = x * (1 - y)
    s = mlab.surf(x, y, z, extent = [0, 1, 0, 1, 0, 1], color = (0.8, 0, 0.8), opacity = 0.8)
    s = mlab.surf(x, y, z2, extent = [0, 1, 0, 1, 0, 1], color = (0.8, 0.8, 0), opacity = 0.8)
    mlab.outline(extent = [0, 1, 0, 1, 0, 1])
    mlab.xlabel('P(H)')
    mlab.ylabel('P(C)')
    mlab.zlabel('P(E = 1)')
    
    mlab.show()
    #cs = contour_surf(x, y, f, contour_z=0)
    return s
test_surf()
from numpy import *
from scipy import optimize
  
def gaussian(height, center_x, center_y, width_x, width_y, offset):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: offset + height*exp(
        -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)
    
def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    med = median(data)
    datam = data - med
    total = datam.sum()
    X, Y = indices(datam.shape)
    x = (X*datam).sum()/total
    y = (Y*datam).sum()/total
    col = datam[:, int(y)]
    width_x = sqrt(abs((arange(col.size)-y)**2*col).sum()/col.sum())
    row = datam[int(x), :]
    width_y = sqrt(abs((arange(row.size)-x)**2*row).sum()/row.sum())
    height = datam.max()
    return height, x, y, width_x, width_y, med
    
def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: ravel(gaussian(*p)(*indices(data.shape)) -
                                    data)
    p, success = optimize.leastsq(errorfunction, params)
    return p
    

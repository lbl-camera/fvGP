def gradient(function, point, epsilon = 10e-3,*args):
    """
    This function calculates the gradient of a function by using finite differences

    Extended description of function.

    Parameters:
    function (function object): the function the gradient should be computed of
    point (numpy array 1d): point at which the gradient should be computed

    optional:
    epsilon (float): the distance used for the evaluation of the function

    Returns:
    numpy array of gradient

    """
    gradient = np.zeros((len(point)))
    for i in range(len(point)):
        new_point = np.array(point)
        new_point[i] = new_point[i] + epsilon
        gradient[i] = (function(new_point,args) - function(point,args))/ epsilon
    return gradient


def hessian(function, point, epsilon = 10e-3, *args):
    """
    This function calculates the hessian of a function by using finite differences

    Extended description of function.

    Parameters:
    function (function object): the function, the hessian should be computed of
    point (numpy array 1d): point at which the gradient should be computed

    optional:
    epsilon (float): the distance used for the evaluation of the function

    Returns:
    numpy array of hessian

    """
    hessian = np.zeros((len(point),len(point)))
    for i in range(len(point)):
        for j in range(len(point)):
            new_point1 = np.array(point)
            new_point2 = np.array(point)
            new_point3 = np.array(point)
            new_point4 = np.array(point)

            new_point1[i] = new_point1[i] + epsilon
            new_point1[j] = new_point1[j] + epsilon

            new_point2[i] = new_point2[i] + epsilon
            new_point2[j] = new_point2[j] - epsilon

            new_point3[i] = new_point3[i] - epsilon
            new_point3[j] = new_point3[j] + epsilon

            new_point4[i] = new_point4[i] - epsilon
            new_point4[j] = new_point4[j] - epsilon

            hessian[i,j] = \
            (function(new_point1,args) - function(new_point2,args) - function(new_point3,args) +  function(new_point4,args))\
            / (4.0*(epsilon**2))
    return hessian

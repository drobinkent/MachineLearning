from numpy import *

epsilon=0.0000001

def batch_gradient_descent(theta_zero_old, theta_one_old,  alpha, dataset,max_iteration):
    theta_zero_new = theta_zero_old
    theta_one_new = theta_one_old
    dataset_size = len(dataset)
    is_converged = False
    
    start_error = sum( [ (theta_zero_new + theta_one_new*dataset[i, 1] - dataset[i, 2])**2 for i in range(dataset_size)] ) 
    number_of_iteration = 0
    while not is_converged:
        number_of_iteration = number_of_iteration +1
        # for each training sample, compute the gradient (d/d_theta j(theta))
        derivative_theta_zero = (1.0/dataset_size) * sum([(theta_zero_new + theta_one_new*dataset[i, 1] - dataset[i, 2]) for i in range(dataset_size)]) 
        derivative_theta_one = (1.0/dataset_size) * sum([(theta_zero_new + theta_one_new*dataset[i, 1] - dataset[i, 2])*dataset[i, 1] for i in range(dataset_size)])

        temp_zero= theta_zero_new - alpha * derivative_theta_zero 
        temp_one=  theta_one_new  - alpha * derivative_theta_one 

        theta_zero_new=temp_zero
        theta_one_new = temp_one

        new_error = sum( [ (theta_zero_new + theta_one_new*dataset[i, 1] - dataset[i, 2])**2 for i in range(dataset_size)] ) 
        if abs(start_error-new_error) <= epsilon:
            print ('Result has converged')
            is_converged = True
        start_error = new_error
        if number_of_iteration ==  max_iteration :
            print("Maximum number of iteration reached")
            is_converged = True
    return theta_zero_new, theta_one_new




    

"""
master function for gradint descent implementation
"""
def gradient_descent():
    dataset = genfromtxt("dataset.csv", delimiter=",")
    alpha = 0.0001
    theta_zero = 1 # initial y-intercept guess
    theta_one = 1 # initial slope guess
    number_of_iterations = 10000000
    
    p,q = batch_gradient_descent(theta_zero,theta_one,alpha,dataset,number_of_iterations)
    print(p,q)
    print("Analytical solution is y = 3.5 + 1.4x")
    print("Solution from gradianet descent algorithm is y = ",p," + ",q,"x")

if __name__ == '__main__':
    gradient_descent()
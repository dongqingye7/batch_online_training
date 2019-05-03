import numpy as np
import matplotlib.pyplot as plt


def sigmoid(X,weights) :  # sigmoid function
    return 1.0/(1.0 + np.exp(-np.dot(X,weights.T)))

def error_func(weights, X, label): 
    
    sigmoid_output = sigmoid(X,weights) 
    label = np.squeeze(label) 
    e1 = label * np.log(sigmoid_output) 
    e2 = (1 - label) * np.log(1 - sigmoid_output) 
    error = -e1 - e2 
    return np.mean(error)

def log_gradient(weights, X, label): 
    a = sigmoid(X,weights) - label 
    label = label.reshape(X.shape[0], -1) 
    gradient_descent = np.dot(a.T, X) 
    return gradient_descent/X.shape[0]

def batch_NN(training,label_training,weights,learning_rate):
  error=error_func(weights,training,label_training)
  converage=0.001
  num_iteration=1

  change_error=np.matrix(np.zeros(100000)).T
  change_error[num_iteration-1]=error

  norm_gradient=np.matrix(np.zeros(100000)).T
  gradient = log_gradient(weights, training, label_training)
  norm_g=np.linalg.norm(gradient,1)
  norm_gradient[num_iteration-1]=norm_g

  while (norm_g>converage) & (num_iteration<100000):
    weights=weights-learning_rate*(gradient)
    error=error_func(weights,training,label_training)
    num_iteration+=1

    change_error[num_iteration-1]=error
    gradient=log_gradient(weights,training,label_training)
    norm_g=np.linalg.norm(gradient,1)
    norm_gradient[num_iteration-1]=norm_g
    
  change_error=change_error[np.where(change_error > 0.0)]
  norm_gradient=norm_gradient[np.where(norm_gradient > 0.0)]
  return weights,num_iteration,change_error,norm_gradient


def pred_values(weights, X): 
  pred = sigmoid(X,weights) 
  pred_value = np.where(pred >= .5, 1, 0) 
  return pred_value

#plot change of training error
def plot_change_of_training_error(change_error, num_iteration): 
    plt.figure(figsize=(6,4))
    x= np.matrix(np.arange(1, num_iteration+1, 1)).T
    change_error = change_error.T
    plt.plot(x[:, 0], change_error[:,0], c='k') 
    plt.title('Changes of Training Error vs. Number of Iteration')
    plt.xlabel('Number of Iteration') 
    plt.ylabel('Change of Training Error') 
    plt.show()   

def plot_norm_of_gradient(norm_gradient, num_iteration): 
    plt.figure(figsize=(6,4))
    x= np.matrix(np.arange(1, num_iteration+1, 1)).T
    norm_gradient = norm_gradient.T
    plt.plot(x[:, 0], norm_gradient[:,0], c='k') 
    plt.title('Changes of Norm of Gradient vs. Number of Iteration')
    plt.xlabel('Number of Iteration') 
    plt.ylabel('Changes of Norm of Gradient') 
    plt.show() 

def plot_data(T1, T2, s): 
    plt.figure(figsize=(6,4))
    plt.scatter([T1[:, 1]], [T1[:, 2]], c='b', label='y = 0') 
    plt.scatter([T2[:, 1]], [T2[:, 2]], c='r', label='y = 1')
    testing = np.vstack((T1, T2)).astype(np.float32)
    x1 = [np.min(testing[:, 0] - 5), np.max(testing[:, 1] + 5)]
    x2 = -(weights[0,0]+np.dot(weights[0,1], x1))/weights[0,2]
    plt.plot(x1, x2, c='k', label='Decision Boundary') 
    plt.title('Testing Data & Trained Decision Boundary')
    plt.xlabel('x1') 
    plt.ylabel('x2') 
    plt.legend() 
    plt.show()


if __name__ == "__main__":
  mean1 = [1, 0]
  mean2=[0,1.5]
  cov1 = [[1,0.75], [0.75,1]]  # diagonal covariance
  cov2=[[1,0.75], [0.75,1]]  # diagonal covariance
  R1= np.random.multivariate_normal(mean1, cov1, 500)
  R2= np.random.multivariate_normal(mean2, cov2, 500)
  T1= np.random.multivariate_normal(mean1, cov1, 500)
  T2= np.random.multivariate_normal(mean2, cov2, 500)

  R1=np.hstack((np.matrix(np.ones(500)).T,R1))
  R2=np.hstack((np.matrix(np.ones(500)).T,R2))
  T1=np.hstack((np.matrix(np.ones(500)).T,T1))
  T2=np.hstack((np.matrix(np.ones(500)).T,T2))
  label_training=np.vstack((np.matrix(np.zeros(500)).T,(np.matrix(np.ones(500)).T)))
  label_testing=np.vstack((np.matrix(np.zeros(500)).T,(np.matrix(np.ones(500)).T)))

  training=np.vstack([R1, R2])
  testing=np.vstack([T1, T2])
  print('---------------------------------------------')
  print ('Perform batch training using gradient descent')
  print('---------------------------------------------')
  
  #initialize weights
  weights=np.matrix(np.zeros(training.shape[1]))
  #set up learning rate
  learning_rate=input('Learning Rate: ')
  learning_rate=float(learning_rate)
  #run batch_NN
  weights,num_iteration,change_error,norm_gradient=batch_NN(training,label_training,weights,learning_rate)
  print('number of Iteration: ',num_iteration)

  label_predict=pred_values(weights,testing)
  accuracy=np.sum(label_testing== label_predict)/1000

  print('Accuracy: ',accuracy)
  plot_data(T1, T2, weights)
  plot_change_of_training_error(change_error,num_iteration)
  plot_norm_of_gradient(norm_gradient, num_iteration)


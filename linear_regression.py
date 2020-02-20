import os
import numpy as np
import json    

#h(x) = theta0 + theta1*x
def cost(x, y, theta):
    # TODO Calculate cost for current model
    n = len(y)
    c = 0
    for i in range(n):
        c = c + pow(theta[0] + theta[1]*x[i] - y[i], 2)
    c = c / (2*n)
    return c

def gradient(x, y, theta):
    # TODO Calculate gradient vector of cost function at current position of theta
    dJ = np.zeros(len(theta))
    n = len(y)
    for i in range(n):
        dJ[0] = dJ[0] + theta[0] + theta[1]*x[i] - y[i]
        dJ[1] = dJ[1] + (theta[0] + theta[1]*x[i] - y[i]) * x[i]
    dJ[0] = dJ[0] / n
    dJ[1] = dJ[1] / n
    return dJ

def gradient_descent(x, y, alpha, init_theta, n_iter):
    # TODO Implement gradient descent algorithm
    theta = init_theta
    i = 0
    j = 0
    fp = open("history.txt", "w")
    for i in range(0, n_iter):
        j = cost(x, y, theta)
        print("%d theta: %f, theta1: %f, cost %.3f" % (i, theta[0], theta[1], j))
        fp.write("%d theta0: %.3f, theta1: %.3f, cost: %.3f\n" % (i, theta[0], theta[1], j))

        dJ = gradient(x, y, theta)
        theta = theta - alpha * dJ
        theta = np.round(1000*theta)/1000
    fp.close()
    return theta, j

def load_data(file_path):
    # TODO Load data from file
    x = []
    y = []
    if os.path.isfile(file_path):
        with open(file_path) as fp:
            while(1):
                line = fp.readline()

                if (not line):
                    break
                
                [s1, s2] = line.split(",")
                x.append(float(s1))
                y.append(float(s2))
        return np.array(x), np.array(y)
    else:
        print("Error! File %s not found!\n" % (file_path))
        exit(0)        

def load_config():
    # TODO Load configurations from file config.json
    if os.path.isfile("config.json"):
        with open("config.json", "r") as fp:
            config = json.load(fp)
            return config
    else:
        print("Error! File config.json not found!\n")
        exit(0)

def main():
    config = load_config()
    #print(config)

    init_theta = config["Theta"]
    alpha = float(config["Alpha"])
    n_iter = int(config["NumIter"])
    #print("Theta =  [%f, %f], Alpha = %f, NumIter = %d" % (theta[0], theta[1], alpha, n_iter))

    x, y = load_data(config["Dataset"])
    #print(x, y)

    gradient_descent(x, y, alpha, init_theta, n_iter)
    
if __name__ == "__main__":
    main()

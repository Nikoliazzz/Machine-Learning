'''
This file is an example of fitting a sin(x) curve using a fifth-degree model.
代码能力比较弱，还请老师批评指正 ^_^
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def plot_curve(X,y,y_pred):
    plt.figure(figsize=(8,6))
    plt.plot(X, y, label='sin(x)', color='blue', linewidth=2)
    plt.plot(X, y_pred, label='Our curve', color='red', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("/root/Y21/Machine_Learning/picture_sin.png")

#Our Linear model
class Our_Linear_Regression(object):
    def __init__(self):
        self.w = np.random.randn(6) 

    def predict(self,X):
        return np.dot(X, self.w)

    def loss_function(self,y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def fit(self,X,y):
        learning_rate = 0.1  
        epochs = 1000000
        n = len(X)    
        loss_history = []
        for i in range(epochs):
            y_pred = self.predict(X) 
            

            loss = self.loss_function(y, y_pred)
            loss_history.append(loss)
            
        
            gradient = -2/n * np.dot(X.T, (y - y_pred)) 
            
        
            self.w -= learning_rate * gradient
            
        
            if i % (epochs // 10) == 0:
                print(f"Epoch {i}, Loss: {loss}")


def main():
    # Data loaded:
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)

    # Normalization:
    std = StandardScaler()
    X = np.vstack([x**i for i in range(6)]).T
    X = std.fit_transform(X)

    #Our lr:    
    model = Our_Linear_Regression()
    model.fit(X,y)
    y_pred = model.predict(X)

    plot_curve(x,y,y_pred)
  

if __name__ == "__main__":
    main()
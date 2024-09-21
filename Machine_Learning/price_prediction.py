'''
This file is an example of using linear regression do the price prediction.
由于使用的数据集包含354维特征,因此只输出了模型拟合后预测值和真实值的对比图。
代码能力比较弱，还请老师批评指正 ^_^
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def plot_pca(train_X, train_label):
    pca = PCA(n_components=2)  
    train_X_pca = pca.fit_transform(train_X)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(train_X_pca[:, 0], train_X_pca[:, 1], c=train_label, cmap='viridis')
    
    plt.title('PCA of the Features')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Target: Price') 
    plt.show()
    plt.savefig('/root/Y21/Machine_Learning/picture_price_PCA.png')

def plot_predictions(train_X, train_label, model):

    predictions = model.predict(train_X)

    plt.figure(figsize=(10, 6))
    plt.scatter(predictions, train_label, s=10)
    plt.plot([min(train_label), max(train_label)], [min(train_label), max(train_label)], linestyle='--', color='red')  # 对角线
    plt.title('True Value vs Predicted Value')
    plt.xlabel('Predicted House Price')
    plt.ylabel('Ture House Price')
    plt.show()
    plt.savefig('/root/Y21/Machine_Learning/picture_price.png')

#Our Linear model
class Our_Linear_Regression(object):
    def __init__(self,n_features):
        self.w = np.random.randn(n_features)* 0.01
        self.b = np.random.randn(1)* 0.01

    
    def predict(self,X):
        return np.dot(X,self.w) + self.b

    def loss_function(self,y,y_pred):
        return np.mean((y -y_pred)**2)
        
    def fit(self,X,y):
        learning_rate=0.01
        epochs=100000
        n = len(y) 
        for i in range(epochs):
            y_pred = self.predict(X) 
            loss = self.loss_function(y,y_pred)

            gradient_w = -2/n * np.dot(X.T, (y - y_pred))
            gradient_b = -2 * np.mean(y - y_pred)

            self.w -= learning_rate * gradient_w
            self.b -= learning_rate * gradient_b
            if i % (epochs // 10) == 0:
                print(f"Epoch {i}, Loss: {loss}")
def main():
    # Data loaded:
    train_path = '/root/Y21/Machine_Learning/kaggle_house_price/house_price_train.csv'
    train_data = pd.read_csv(train_path).values

    train_label = train_data[:, -1]
    train_X = train_data[:, :-1]
    print(train_X.shape)

    # Normalization:
    std = StandardScaler()
    train_X = std.fit_transform(train_X)
    num_features = train_X.shape[1]



    #Our lr:
    model = Our_Linear_Regression(num_features)
    model.fit(train_X, train_label)
    plot_predictions(train_X, train_label,model)


if __name__ == "__main__":
    main()




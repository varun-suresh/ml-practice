import numpy as np
import random
class KMeans:
    def __init__(self):
        pass
    
    def fit(self,X:np.ndarray, k:int,max_iters=100):
        self.X = X
        self.k = k
        self.centers = X[np.random.choice(X.shape[0],k,replace=False)]

        for iteration in range(max_iters):
            distances = np.linalg.norm(X[:,np.newaxis]-self.centers,axis=2)
            labels = np.argmin(distances,axis=1)
            new_centers = []
            for i in range(k):
                new_centers.append(np.mean(X[labels==i],axis=0))
            new_centers = np.array(new_centers)
            if np.allclose(new_centers,self.centers):
                break
            self.centers = new_centers
         
        print(f"Fit complete")
    
    def predict(self,X:np.ndarray) -> np.ndarray:
        res = np.argmin(np.linalg.norm(X[:,np.newaxis] - self.centers,axis=2),axis=1)
        return res
    
    def compute_loss(self,X:np.ndarray) -> np.ndarray:
        return np.sum(np.min(np.linalg.norm(X[:,np.newaxis] - self.centers,axis=2) ** 2,axis=1))
        
        
                

if __name__ == "__main__":
    X = np.array([
    [1, 2], [1, 4], [1, 0],
    [10, 2], [10, 4], [10, 0]
    ])

    kmeans = KMeans()
    kmeans.fit(X, k=2)
    print(kmeans.predict(X))
    print(kmeans.compute_loss(X))
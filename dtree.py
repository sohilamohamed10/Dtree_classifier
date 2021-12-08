
import numpy as np
from operator import itemgetter
import math

class CART():

	def  __init__(self,method="gini"):

		self.All_features_values=[]
		self.All_impurity=[]
		self.features_impurity=[]
		self.root = None
		self.left=None
		self.right=None
		self.method=method
		self.label=None
		self.best_feature=None
		self.best_value=None

	def fit (self,X_train,y_train):
		self.root=CART(self.method)
		self.root.build_tree(X_train,y_train)
 
	def build_tree(self,X_train,y_train):

		if len(np.unique(y_train)) == 1:  #condition to terminate
			self.label = y_train[0]
			return

		self.categorize(X_train)
		for i in range(len(self.All_features_values)):      # loop of features
			
			value_impurity=[]         #unique value/category gini index or entropy
			w_impurity=0                         
			p_prop=list(y_train).count(1)/X_train.shape[0]           #+ve and -ve classes propabiliies for whole feature
			n_prop=list(y_train).count(0)/X_train.shape[0]
			entropy=-(p_prop* np.log2(p_prop))-(n_prop* np.log2(n_prop))     # entropy of whole dataset 

			for value in range(len(self.All_features_values[i])):           
				value_indices=list(np.where(X_train[:,i]==self.All_features_values[i][value])[0])
				if (len(value_indices)==1):
					value_impurity.append(0)
				else:
					classes=list(itemgetter(*value_indices)(y_train)) 
					p_p=classes.count(1)/len(value_indices)
					n_p=classes.count(0)/len(value_indices)
					value_impurity=self.impurity(value_impurity,value_indices,entropy,p_p,n_p)

				w_impurity=w_impurity+value_impurity[0]*(len(value_indices)/X_train.shape[0])

			self.All_impurity.append(value_impurity)
			if self.method=="gini":
				self.features_impurity.append(w_impurity)
				
			else:
				self.features_impurity.append(w_impurity+entropy)
		self.best_parameters(X_train,y_train)

	def best_parameters(self,X_train,y_train):
		if (self.method=="gini"):
			self.best_feature=self.features_impurity.index(min(self.features_impurity))
		else:
			self.best_feature=self.features_impurity.index(max(self.features_impurity))

		best_value_idx=self.All_impurity[self.best_feature].index(min(self.All_impurity[self.best_feature]))
		self.best_value=self.All_features_values[self.best_feature][best_value_idx]
		print(self.best_feature,self.best_value)
		self.split(X_train,y_train)


	def split(self,X_train,y_train):
		temp_data=np.concatenate((X_train, y_train[:,None]), axis=1)
		if (len(self.All_features_values[self.best_feature])>=3):
			selected_data_l= temp_data[temp_data[:, self.best_feature] <= self.best_value]
			selected_data_r= temp_data[temp_data[:, self.best_feature] > self.best_value]
		else:
			selected_data_l= temp_data[temp_data[:, self.best_feature] == self.best_value]
			selected_data_r= temp_data[temp_data[:, self.best_feature] != self.best_value]
		
		feature_left=selected_data_l[:,:-1]
		label_left=selected_data_l[:,-1] 
		self.left = CART()
		self.left.build_tree(feature_left,label_left)

		feature_right=selected_data_r[:,:-1]
		label_right=selected_data_r[:,-1] 
		self.right = CART()
		self.right.build_tree(feature_right,label_right)
	
	def impurity (self,value_impurity,value_indices,entropy,p_p,n_p):

		if (self.method=="gini"):
			value_impurity.append(1-((p_p)**2+(n_p)**2))
		else:
			value_impurity.append(-(p_p * np.log2(p_p))-(n_p * np.log2(n_p)))

		return value_impurity

	def categorize(self,X_train):
		for i in range (X_train.shape[1]): 
			temp=[]
			values=np.unique(X_train[:,i]).tolist()
			if (len(values)<=3):    
				self.All_features_values.append(values)     #list of lists(features*unique values of each feature)
			else:
				values.sort()
				threshold_idx=math.ceil(len(values)/10)
				for v in range(threshold_idx,len(values),threshold_idx):
					temp.append(values[v])
				temp.append(values[-1]) 
				self.All_features_values.append(temp)

	def predict(self,X_test):
	    for f in X_test :
	    	return [self.root.predict2(f)]

	def predict2(self,f):
		if self.best_feature != None:
			if f[self.best_feature] <= self.best_value:
				return self.left._predict(f)                  
			else:
				return self.right._predict(f)
		else: 
			return self.label
	def evaluate(self,pred,y_test):
		
		return (sum(pred == y_test) / len(y_test))*100



#example---------------------------------------------------------------------------
file = open("cardio_train.csv")
data_set = np.loadtxt(file, delimiter=";",skiprows=1,dtype=int)
x=data_set[:,1:-1]
y=data_set[:,-1]
X_train, X_test, y_train, y_test = x[:63000], x[63000:], y[:63000], y[63000:]
cart=CART(method="gini")
cart.fit(X_train,y_train)
predictions=cart.predict(X_test)
accuracy=cart.evaluate(predictions,y_test)


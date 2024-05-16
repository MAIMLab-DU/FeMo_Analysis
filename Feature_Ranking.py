# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:27:23 2024

@author: Monaf Chowdhury

This class will be used to rank top features.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.feature_selection import SelectFromModel, RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, RandomForestClassifier 
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# https://scikit-learn.org/stable/modules/feature_selection.html Check out this page for some guidelines

class FeatureRanker:
    def __init__(self, X, Y):
        """
        @author: Moniruzzaman Akash
        @co-author: Monaf Chowdhury 
        Initialize the class with X and Y values for feature ranking
        Parameters
        ----------
        X : Numpy.ndarray. Shape(number of samples, number of features)
            Normalized form of X as input. X has already features extracted and normalized
        Y : numpy.ndarray. Shape(number of samples)
            Labels of shapes according to samples
        Returns
        -------
        None.
        """
        self.X = X
        self.Y = Y

    def random_forest_ranking(self, n):
        # Incomplete/doesn't give good scores. need optimization.
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
        clf.fit(self.X, self.Y)
        feature_importances = clf.feature_importances_
        top_n_indices = np.argsort(feature_importances)[-n:][::-1]
        return top_n_indices

    def pca_ranking(self, n):
        # Incomplete
        pca = PCA(n_components=n)
        pca.fit(self.X)
        top_n_indices = np.argsort(np.abs(pca.components_))[0][-n:][::-1]
        return top_n_indices

    def nca_ranking(self, n):
        """
        @author: Monaf Chowdhury, Moniruzzaman Akash
        Generates the NCA of shape (n,number of features). Here N is the number of components. Lower the index of the component 
        the better it can explain the correlation with its features. Therefore, [0] th indexed component is chosen for feature 
        extraction.

        Parameters
        ----------
        n : int. Number of component. Ideally 30
            The number of components we want for genearting NCA.

        Returns
        -------
        nca_top_n : numpy.ndarray. Shape(n)
            An array of 'n' most important indices 
        """
        print("Neighbourhood component analysis is going on... \n")
        nca = NeighborhoodComponentsAnalysis(n_components=30, init = 'auto', max_iter = 1000, tol = 1e-5, verbose = 1, random_state = 0)
        nca.fit(self.X, self.Y)
        nca_top_n = np.argsort(np.abs(nca.components_))[0][-n:][::-1]
        print(f"NCA top features: {nca_top_n}\n")
        return nca_top_n
    
    def xgboost_ranking(self, n):
        """
        @author: Moniruzzaman Akash, Monaf Chowdhury 
        
        Parameters
        ----------
        n : int. Usually 30
            The number of features that are desired for feature selection. 

        Returns
        -------
        xgb_top_n :  numpy.ndarray. Shape(n)
            An array of 'n' most important indices 

        """
        print("XGBoost is going on... \n")
        xgb = XGBClassifier(learning_rate=0.01, n_estimators=100, max_depth=3, random_state=0, booster = 'gbtree')
        xgb.fit(self.X, self.Y)
        feature_importances = xgb.feature_importances_
        xgb_top_n = np.argsort(feature_importances)[-n:][::-1]
        print(f"XGBoost top features: {xgb_top_n}\n")
        return xgb_top_n

    def logistic_regression_ranking(self, n):
        """
        @author: Monaf Chowdhury, Moniruzzaman Akash
        L1 regularization penalizes the absolute value of the coefficients, leading to sparse solutions where irrelevant 
        features have zero coefficients. 
        
        Parameters
        ----------
        n : int. Usually 30
            The number of features that are desired for feature selection. 

        Returns
        -------
        l1_based_top_n : numpy.ndarray. Shape(n)
            An array of 'n' most important indices 
            
        """
        print("L1 based feature selection is going on... \n")
        
        log_reg = LogisticRegression(penalty='l1', solver='liblinear', random_state = 42)     
        log_reg.fit(self.X, self.Y)
        l1_based_top_n = SelectFromModel(log_reg, prefit=True).get_support(indices=True)
        print(f"L1 based top features: {l1_based_top_n}\n")
        
        return l1_based_top_n
    
    def recursive_feature_elimination(self,n):
        """
        @author: Monaf Chowdhury
        Recursively eliminate the least important features based on model coefficients or feature importance scores.
        
        Parameters
        ----------
        n : int. Usually 30
            The number of features that are desired for feature selection. 

        Returns
        -------
        recursive_top_n : numpy.ndarray. Shape(n)
            An array of 'n' most important indices 

        """
        print("Recursive feature elimination is going on... \n")
        
        # estimator = ExtraTreesClassifier()    # ExtraTreesClassifier() gives slightly low f1 score.
        estimator = AdaBoostClassifier()
        rfe = RFE(estimator=estimator, n_features_to_select = n, step=1, verbose = 1)
        rfe.fit(self.X, self.Y)
        recursive_top_n = rfe.get_support(indices= True)
        
        print(f"Recursive feature elimination top features: {recursive_top_n}\n")
        
        return recursive_top_n
    
    def ensemble_feature_selection(self, fusion_criteria, nca_top_n, xgb_top_n, l1_based_top_n, recursive_top_n ):
        """
        @author: Monaf Chowdhury
        Ensembles the features based on user defined policy and finds out the best 'k' number of feautres
        Parameters
        ----------
        fusion_criteria : int. Usually 3
            Atleast how many number of feature selection methods must have common features

        Returns
        -------
        selected_features : numpy.ndarray. Shape(k)
            An array of 'k' most important indices 

        """
        # nca_top_n = self.nca_ranking(n)
        # xgb_top_n = self.xgboost_ranking(n)
        # l1_based_top_n = self.logistic_regression_ranking(n)
        # recursive_top_n = self.recursive_feature_elimination(n)
        # fusion_criteria = int(input("Atleast how many number of feature selection methods must have common features: "))
        selected_features = []
        if fusion_criteria == 4:
            # Common elements between NCA, XGBoost, L1 Based, Recursive 
            common_elements = np.intersect1d(nca_top_n, np.intersect1d(xgb_top_n, np.intersect1d(l1_based_top_n, recursive_top_n))) # Finding out the common elements 
            selected_features.append(common_elements)
        elif fusion_criteria == 3: 
            # Common elements between three methods 
            # Common elements between NCA, L1 Based, Recursive 
            common_elements = np.intersect1d(nca_top_n, np.intersect1d(l1_based_top_n, recursive_top_n))
            selected_features.append(common_elements)
            
            # common elements between XGBoost, L1 Based, Recursive 
            common_elements = np.intersect1d(xgb_top_n, np.intersect1d(l1_based_top_n, recursive_top_n))  
            selected_features.append(common_elements)
            
            # common elements between NCA, XGBoost, L1 Based
            common_elements = np.intersect1d(nca_top_n, np.intersect1d(xgb_top_n, l1_based_top_n)) 
            selected_features.append(common_elements)
            
            # common elements between NCA, XGBoost, Recursive
            common_elements = np.intersect1d(nca_top_n, np.intersect1d(xgb_top_n, recursive_top_n))
            selected_features.append(common_elements)
        
        elif fusion_criteria == 2: 
            # Common elements between two methods 
            # common elements between NCA and XGBoost
            common_elements = np.intersect1d(nca_top_n, xgb_top_n) 
            selected_features.append(common_elements)

            # common elements between NCA and L1 Based
            common_elements = np.intersect1d(nca_top_n, l1_based_top_n) 
            selected_features.append(common_elements)

            # common elements between NCA and Recursive
            common_elements = np.intersect1d(nca_top_n, recursive_top_n) 
            selected_features.append(common_elements)

            # common elements between XGB and Recursive
            common_elements = np.intersect1d(xgb_top_n, recursive_top_n)
            selected_features.append(common_elements)

            # common elements between XGB and L1 based
            common_elements = np.intersect1d(xgb_top_n, l1_based_top_n)  
            selected_features.append(common_elements)

            # common elements between L1 Based and Recursive based
            common_elements = np.intersect1d(l1_based_top_n, recursive_top_n)
            selected_features.append(common_elements)
            
        else:
            # Selected all methods
            selected_features = [nca_top_n, xgb_top_n, l1_based_top_n, recursive_top_n]
            
        selected_feature_with_overlapping_elements = np.concatenate(selected_features)
        selected_features =  np.unique(selected_feature_with_overlapping_elements)
        return selected_features
    
    
    def f_classif_ranking(self, n):
        " Incomplete"
        selector = SelectKBest(score_func=f_classif, k=n)
        selector.fit(self.X, self.Y)
        top_n_indices = np.argsort(selector.scores_)[-n:][::-1]
        return top_n_indices
    
    def use_matlab_nca_features(self):
        #1 indexed top features from Matlab
        index_top_features = np.array([90,67,70,19,83,54,55,35,51,71,22,7,1,39,60,3,44,28,21,87,6,57,74,23,38,86,69,18,58,20])
        print("Using Matlab generated Top features\n")
        # print("Top 30 feature Indexes(1 indexed):\n", index_top_features)
        index_top_features = index_top_features-1 #Make it zero indexed for python
        return index_top_features
        
    def old_nca_ranking(self):
        # FEATURE RANKING BY NEIGHBOURHOOD COMPONENT ANALYSIS (NCA) ==============


        #This is to tune the regularization parameter.
        #Tuning the regularization parameter helps to correctly detect 
        #the relevant features in the data.


        # Finding the best lambda using K-fold cross-validation
        K = 5  # Number of folds
        p = 0.19  # fraction of observation in test set

        np.random.seed(0)

        # cvp = StratifiedKFold(n_splits=K)  # Stratified K-fold division

        #Split the data into training and testing sets using train_test_split
            #stratified split ensures the target class distribution is preserved in both the training and testing sets
        Xtrain, Xtest, ytrain, ytest = train_test_split(self.X, self.Y, test_size=p, random_state=42, stratify=self.Y)


        lambdavalues = np.linspace(0.000000000000000001, 2, 20) / len(self.Y)
        lossvalues = []

        for lambda_value in lambdavalues:
            # Create the NCA model with the current lambda value
            nca = NeighborhoodComponentsAnalysis(n_components=None,random_state=0,verbose=1, max_iter=50, tol=1e-5)
            
            # Transform the training data using NCA
            X_train_transformed = nca.fit_transform(Xtrain, ytrain)

            # Transform the test data using NCA
            X_test_transformed = nca.transform(Xtest)

            # Create the Ridge regression model with alpha = lambda_value
                #we use Ridge regression (L2 regularization) as 
                #it is suitable for mean squared error (MSE) loss
            model = Ridge(alpha=1/lambda_value, solver='lbfgs', positive = True)

            # Train the model on the transformed training data
            model.fit(X_train_transformed, ytrain)

            # Make predictions on the transformed test data
            y_pred = model.predict(X_test_transformed)

            # Calculate the Root mean squared error (RMSE) loss
            loss = np.sqrt(mean_squared_error(ytest, y_pred))

            # Store the loss value for this lambda
            lossvalues.append(loss)
            
            
            
            # Plot lambda vs. loss
            plt.figure()
            plt.plot(lambdavalues, lossvalues, 'ro-')
            plt.xlabel('Lambda values')
            plt.ylabel('Loss values')
            plt.grid(True)
            plt.show()
            
            idx = np.argmin(lossvalues) # Find the index of minimum loss.
            bestlambda = lambdavalues[idx]  # Find the best lambda value.
            
            
            print("Best lambda: ", bestlambda)


        #~~~~~~~~~~~ Use the selected lambda to find optimum NCA model~~~~~~~~~~~%

        # Applying the best model
        # Create the NCA model with the current lambda value
        ncaMdl = NeighborhoodComponentsAnalysis(n_components=None, random_state=0,verbose=1)

        # Transform the training data using NCA
        X_train_transformed = ncaMdl.fit_transform(self.X, self.Y)


        # Create the Ridge regression model with alpha = 1/bestlambda
            #we use Ridge regression (L2 regularization) as 
            #it is suitable for mean squared error (MSE) loss
        model_final = Ridge(alpha=1/bestlambda, solver='lbfgs', positive = True)

        # Train the model on the transformed training data
        model_final.fit(X_train_transformed, self.Y)



        # Plot the feature weights
        plt.semilogx(model_final.coef_, 'ro')
        # plt.semilogx(nca.weights_, 'ro')
        plt.xlabel('Feature index')
        plt.ylabel('Feature weight')
        plt.grid(True)
        plt.show()


        # ~~~~~~~~~~~~~~~Extract the feature ranking information~~~~~~~~~~~~~~

        #Creates a vector with feature index numbers starting with 0. Matlab starts with 1
        feature_index = np.arange(0, self.X.shape[1])

        feature_ranking = np.column_stack((feature_index, model_final.coef_)) #% Combines feature index and weights in a matrix
        I_sort = np.argsort(feature_ranking[:, 1])[::-1]  # Sort in descending order
        feature_ranking = feature_ranking[I_sort, :] #Apply the sorted index to sort the features

        #Number of significant features to consider
        n_top_features_to_consider = 30



        index_top_features = feature_ranking[:n_top_features_to_consider, 0].astype(int)
        print("Top 30 feature Indexes(1 indexed):\n", index_top_features+1)
        
        return index_top_features
        
        
        
if __name__ == '__main__':
    # Example usage
    X = np.random.rand(100, 10)  # Replace with your dataset
    y = np.random.randint(0, 2, 100)  # Replace with your target labels

    feature_ranker = FeatureRanker(X, y)
    n = 5  # Number of top features to select

    rf_top_n = feature_ranker.random_forest_ranking(n)
    pca_top_n = feature_ranker.pca_ranking(n)
    nca_top_n = feature_ranker.nca_ranking(n)
    xgb_top_n = feature_ranker.xgboost_ranking(n)
    lr_top_n = feature_ranker.logistic_regression_ranking(n)
    f_classif_top_n = feature_ranker.f_classif_ranking(n)

    print("Random Forest Top Features:", rf_top_n)
    print("PCA Top Features:", pca_top_n)
    print("NCA Top Features:", nca_top_n)
    print("XGBoost Top Features:", xgb_top_n)
    print("Logistic Regression Top Features:", lr_top_n)
    print("F-Classif Top Features:", f_classif_top_n)



    

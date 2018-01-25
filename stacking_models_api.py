from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNetCV
from cross_valid_api import root_mean_squred_error
from sklearn.decomposition import PCA

class StackingAveragedModels(BaseEstimator, TransformerMixin, RegressorMixin):
    def __init__(self, 
        sl_base_models_dict = None, 
        semi_sl_base_models_dict = None, 
        usl_base_models_dict = None,
        meta_model = None,
        output_nums = 1,
        n_folds=5, target_col = None, eval_func = root_mean_squred_error, random_state = 0,
        is_classification = False):
        
        self.n_folds = n_folds
        self.target_col = 'target'
        if target_col is not None:
            self.target_col = target_col
        
        self.eval_func = eval_func 
        self.random_state = random_state
        self.is_classification = is_classification
        
        self.base_models_info = {
            'supervised': {
                'base models': sl_base_models_dict,
                '_fit_func' : self._supervised_stacked_fit,
                '_estimators' : {}, # key: model name, data: a list of out-of-fold fitted instances
                '_predict_func': self._supervised_predict
            },
            'semi-supervised': {
                'base models': semi_sl_base_models_dict,
                '_fit_func' : self._semi_supervised_stacked_fit,
                '_estimators' : {}, # key: model name, data: a list of out-of-fold fitted instances
                '_predict_func': self._semi_supervised_predict
            },
            'unsupervised': {
                'base models': usl_base_models_dict,
                '_fit_func' : self._unsupervised_stacked_fit,
                '_estimators' : {}, # key: model name, data: fitted instances (no out-of-fold predictions)
                '_predict_func': self._unsupervised_predict
            }
        }
        self.reset_meta_model(meta_model)
        self.out_of_fold_predictions = pd.DataFrame()
        self.fitted = False
        self.output_nums = output_nums
        
    def reset_meta_model(self, meta_model):
        self.meta_model = meta_model
    
    def fit(self, X, y):
        for ml_method in self.base_models_info:
            fit_func = self.base_models_info[ml_method]['_fit_func']
            base_model_dict = self.base_models_info[ml_method]['base models']
            if base_model_dict is not None:
                fit_func(X, y)
                self.fitted = True
        
    # Fit the data on clones of the original models, 
    # return generated out-of-fold predictions of training set as meta features to train on
    # and meta features of test set
    # For supervised learning with fit predict methods
    def _supervised_stacked_fit(self, train_x, train_y):
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        base_models_dict = self.base_models_info['supervised']['base models']
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        for model_name, model in base_models_dict.items():
            print("\n==================\n", model_name)
            
            score_sum = .0
            out_of_fold_predictions = np.zeros((train_x.shape[0],))
            base_models = []
            
            for train_index, holdout_index in kfold.split(train_x, train_y):
                instance = copy.deepcopy(model)
                instance.fit(train_x.loc[train_index], train_y[train_index])
                
                y_pred = self._get_prediction(instance, train_x.loc[holdout_index])
                score = self.eval_func(train_y[holdout_index], y_pred)
                score_sum += score
                print("score=", score)
                
                out_of_fold_predictions[holdout_index] = y_pred
                base_models.append(instance)
                
            print("Avg score = ", score_sum/self.n_folds)
            
            self.out_of_fold_predictions[model_name] = out_of_fold_predictions
            self.base_models_info['supervised']['_estimators'][model_name] = base_models
            
        self.out_of_fold_predictions[self.target_col] = train_y
        
        # return the dataframe for meta features to train and test
        return self.get_meta_train_dataframe()
    
    # for semi supervised methods: knn regressor with fit, kneighbors and predict methods
    # it used the whole dataset instead of out-of-fold predictions
    # output features: knn regressor's result + distances to n-nearest neighbors
    def _semi_supervised_stacked_fit(self, train_x, train_y):
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        base_models_dict = self.base_models_info['semi-supervised']['base models']
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        for model_name, model in base_models_dict.items():
            print("\n==================\n", model_name)
            
            score_sum = .0
            out_of_fold_predictions = np.zeros((train_x.shape[0],))
            out_of_fold_predictions_dist = np.zeros((train_x.shape[0],))
            base_models = []
            
            for train_index, holdout_index in kfold.split(train_x, train_y):
                instance = copy.deepcopy(model)
                instance.fit(train_x.loc[train_index], train_y[train_index])
                
                y_pred = self._get_prediction(instance, train_x.loc[holdout_index]) 
                distances, _ = instance.kneighbors(train_x.loc[holdout_index]) # get n nearest neighbors distances
                out_of_fold_predictions[holdout_index] = y_pred
                out_of_fold_predictions_dist[holdout_index] = np.array(distances).mean(axis=1)
                
                score = self.eval_func(train_y[holdout_index], y_pred)
                score_sum += score
                print("score=", score)
                
                base_models.append(instance)
  
            print("Avg score = ", score_sum/self.n_folds)
            
            self.out_of_fold_predictions[model_name] = out_of_fold_predictions
            self.out_of_fold_predictions[model_name + '_dist'] = out_of_fold_predictions_dist
            self.base_models_info['semi-supervised']['_estimators'][model_name] = base_models
            
        self.out_of_fold_predictions[self.target_col] = train_y
        
        # return the dataframe for meta features to train and test
        return self.get_meta_train_dataframe()
        
    # for unsupervised methods: clustering methods with fit and predict methods
    # it used the whole dataset instead of out-of-fold predictions
    # dummies (one-hot encodings) for clustering labels
    def _unsupervised_stacked_fit(self, train_x, train_y):
        base_models_dict = self.base_models_info['unsupervised']['base models']
        
        for model_name, model in base_models_dict.items():
            print("\n==================\n", model_name)
            
            instance = copy.deepcopy(model)
            instance.fit(train_x)
                
            y_pred = instance.predict(train_x)    
            self.out_of_fold_predictions[model_name] = y_pred
            self.base_models_info['unsupervised']['_estimators'][model_name] = instance
        
        self.out_of_fold_predictions[self.target_col] = train_y
        
        # return the dataframe for meta features to train and test
        return self.get_meta_train_dataframe()
    
    def _get_prediction(self, model, X):
        if self.is_classification is True:  
            # return the probability of being true
            return model.predict_proba(X)[:,1]
        else:
            return model.predict(X)
            
    def get_meta_train_dataframe(self, get_dummies=False, pca_enabled=False, pca_variance_th=1e-4):
        
        ret_df = self.out_of_fold_predictions
        
        if get_dummies is True:
            base_model_dict = self.base_models_info['unsupervised']['base models']
            estimators = self.base_models_info['unsupervised']['_estimators']
            
            # configured and fit to the data already
            if base_model_dict is not None and estimators != {}:
                model_names = [model_name for model_name, _ in base_model_dict.items()]
                ret_df = pd.get_dummies(ret_df, columns=model_names)
                
        if pca_enabled is True:
            pca = PCA(whiten=True)
            features = ret_df.columns.tolist()
            features.remove(self.target_col)
            
            pca.fit(ret_df[features])
            new_n_components = pd.Series(pca.explained_variance_ >= pca_variance_th).value_counts(sort=False)[1]
            print("New features preserved:", new_n_components)
            
            pca = PCA(whiten=True, n_components=new_n_components)
            new_data = np.array(pca.fit_transform(ret_df[features]))
            columns = ['f_{}'.format(i) for i in range(new_n_components)]

            ret_df = pd.DataFrame(data=new_data, columns=columns)
            ret_df[self.target_col] = self.out_of_fold_predictions[self.target_col]
            
        return ret_df
        
    def save_to_csv(self, prefix):
        self.out_of_fold_predictions.to_csv(prefix + '_meta_train.csv', float_format='%.6f', encoding = 'utf-8', index=False)

    def _predict_preprocess(self, X):
        meta_X = pd.DataFrame()
        out_of_fold_predictions = copy.deepcopy(self.out_of_fold_predictions)
        
        for ml_method in self.base_models_info:
            predict_func = self.base_models_info[ml_method]['_predict_func']
            base_model_dict = self.base_models_info[ml_method]['base models']
            estimators = self.base_models_info[ml_method]['_estimators']
            
            if base_model_dict is not None and estimators != {}:
                out_of_fold_predictions, meta_X = predict_func(meta_X, X, out_of_fold_predictions)
                
            
        instance = copy.deepcopy(self.meta_model)
        features = list(out_of_fold_predictions.columns)
        features.remove(self.target_col)
        
        x = out_of_fold_predictions[features]
        y = out_of_fold_predictions[self.target_col]
        instance.fit(x, y)
        
        pred_y = self._get_prediction(instance, x)
        print("meta model's training set score= ", self.eval_func(y, pred_y), "\n")
        
        return instance, meta_X
        
    def predict(self, X):
        if self.fitted is False:
            if self.output_nums == 1:
                shape = (X.shape[0],)
            else:
                shape = (X.shape[0], self.output_nums)
                
            return np.zeros(shape, dtype=np.int)
            
        instance, meta_X = self._predict_preprocess(X)
        return instance.predict(meta_X)
    
    def predict_proba(self, X):
        if self.fitted is False:
            # self.output_nums = 1: binary classification
            if self.output_nums == 1:
                shape = (X.shape[0], self.output_nums+1)
            else:
                shape = (X.shape[0], self.output_nums)
                
            return np.zeros(shape)
        instance, meta_X = self._predict_preprocess(X)
        return instance.predict_proba(meta_X)
        
    def _supervised_predict(self, meta_X, X, out_of_fold_predictions):
        estimator_dict = self.base_models_info['supervised']['_estimators']
        base_models_dict = self.base_models_info['supervised']['base models']
        
        for model_name, _ in base_models_dict.items():
            test_set_predictions = []
            
            for estimator in estimator_dict[model_name]: # each estimator is trained on the partial data
                test_set_predictions.append(self._get_prediction(estimator,X))
            
            meta_X[model_name] = np.column_stack(test_set_predictions).mean(axis=1)
        
        return out_of_fold_predictions, meta_X
    
    def _semi_supervised_predict(self, meta_X, X, out_of_fold_predictions):
        estimator_dict = self.base_models_info['semi-supervised']['_estimators']
        base_models_dict = self.base_models_info['semi-supervised']['base models']
        
        for model_name, _ in base_models_dict.items():
            test_set_predictions = []
            test_set_predictions_dist = []
            
            for estimator in estimator_dict[model_name]: # each estimator is trained on the partial data
                y_pred_test = self._get_prediction(estimator,X)   
                test_distances, _ = estimator.kneighbors(X) # get n nearest neighbors distances
                test_set_predictions.append(y_pred_test)
                test_set_predictions_dist.append(np.array(test_distances).mean(axis=1))
            
            meta_X[model_name] = np.column_stack(test_set_predictions).mean(axis=1)
            meta_X[model_name + '_dist'] = np.column_stack(test_set_predictions_dist).mean(axis=1)
            
        return out_of_fold_predictions, meta_X
        
    def _unsupervised_predict(self, meta_X, X, out_of_fold_predictions):
        estimator_dict = self.base_models_info['unsupervised']['_estimators']
        base_models_dict = self.base_models_info['unsupervised']['base models']
        
        for model_name, _ in base_models_dict.items():
            meta_X[model_name] = estimator_dict[model_name].predict(X)
        
        length = out_of_fold_predictions.shape[0]
        model_names = [model_name for model_name, _ in base_models_dict.items()]
        
        all_df = pd.concat([out_of_fold_predictions, meta_X], join="inner")
        all_df = pd.get_dummies(all_df, columns=model_names)
        out_of_fold_predictions = all_df[:length]
        meta_X = all_df[length:]
        
        out_of_fold_predictions[self.target_col] = self.out_of_fold_predictions[self.target_col]
        
        return out_of_fold_predictions, meta_X


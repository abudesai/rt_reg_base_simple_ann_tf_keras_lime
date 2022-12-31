
import numpy as np, pandas as pd
import sys 
from abc import ABC, abstractmethod
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class DropNATransformer(BaseEstimator, TransformerMixin):  
    ''' Scale ratings '''
    def __init__(self, cols_list): 
        super().__init__()
        self.cols_list = cols_list


    def fit(self, X, y=None): return self
        

    def transform(self, data):    
        if data.empty: return data        
        data = data.dropna(subset=self.cols_list)
        return data


class TypeCaster(BaseEstimator, TransformerMixin):  
    def __init__(self, vars, cast_type):
        super().__init__()
        self.vars = vars
        self.cast_type = cast_type
        
    def fit(self, X, y=None): return self
        

    def transform(self, data):  
        data = data.copy()
        applied_cols = [col for col in self.vars if col in data.columns] 
        for var in applied_cols: 
            data[var] = data[var].apply(self.cast_type)
        return data


class StringTypeCaster(TypeCaster):  
    ''' Casts categorical features as object type if they are not already so.
    This is needed when some categorical features have values that can inferred as numerical.
    This causes an error when doing categorical feature engineering. 
    '''
    def __init__(self, cat_vars): 
        super(StringTypeCaster, self).__init__(cat_vars, str)


class FloatTypeCaster(TypeCaster):  
    ''' Casts float features as object type if they are not already so.
    This is needed when some categorical features have values that can inferred as numerical.
    This causes an error when doing categorical feature engineering. 
    '''
    def __init__(self, num_vars):
        super(FloatTypeCaster, self).__init__(num_vars, float)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select only specified columns."""
    def __init__(self, columns, selector_type='keep'):
        self.columns = columns
        self.selector_type = selector_type
        
        
    def fit(self, X, y=None):
        return self
    
    
    def transform(self, X):
        if self.selector_type == 'keep':
            retained_cols = [col for col in X.columns if col in self.columns]
            X = X[retained_cols].copy()
        elif self.selector_type == 'drop':
            dropped_cols = [col for col in X.columns if col in self.columns]  
            X = X.drop(dropped_cols, axis=1)      
        else: 
            raise Exception(f'''
                Error: Invalid selector_type. 
                Allowed values ['keep', 'drop']
                Given type = {self.selector_type} ''')
        
        return X
    
    

class OneHotEncoderMultipleCols(BaseEstimator, TransformerMixin):  
    def __init__(self, ohe_columns, max_num_categories=10): 
        super().__init__()
        self.ohe_columns = ohe_columns
        self.max_num_categories = max_num_categories
        self.top_cat_by_ohe_col = {}
        
        
    def fit(self, X, y=None):    
        for col in self.ohe_columns:
            if col in X.columns: 
                self.top_cat_by_ohe_col[col] = [ 
                    cat for cat in X[col].value_counts()\
                        .sort_values(ascending = False).head(self.max_num_categories).index
                    ]         
        return self
    
    
    def transform(self, data): 
        data.reset_index(inplace=True, drop=True)
        df_list = [data]
        cols_list = list(data.columns)
        for col in self.ohe_columns:
            if len(self.top_cat_by_ohe_col[col]) > 0:
                if col in data.columns:                
                    for cat in self.top_cat_by_ohe_col[col]:
                        col_name = col + '_' + cat
                        # data[col_name] = np.where(data[col] == cat, 1, 0)
                        vals = np.where(data[col] == cat, 1, 0)
                        df = pd.DataFrame(vals, columns=[col_name])
                        df_list.append(df)
                        
                        cols_list.append(col_name)
                else: 
                    raise Exception(f'''
                        Error: Fitted one-hot-encoded column {col}
                        does not exist in dataframe given for transformation.
                        This will result in a shape mismatch for train/prediction job. 
                        ''')
        transformed_data = pd.concat(df_list, axis=1, ignore_index=True) 
        transformed_data.columns =  cols_list
        return transformed_data


class CustomScaler(BaseEstimator, TransformerMixin, ABC): 
    def __init__(self, cols_list):
        super(CustomScaler, self).__init__()
        self.cols_list = cols_list
        self.scaler = None  # defined in derived class
        
        
    def fit(self, data): 
        sub_data = data[self.cols_list]
        self.scaler.fit(sub_data)
        return self
    
    
    def transform(self, data): 
        
        data.reset_index(inplace=True, drop=True)
        
        other_cols = [col for col in data.columns if col not in self.cols_list]
        other_data = data[other_cols]
        
        sub_data = data[self.cols_list]
                
        scaled_data = self.scaler.transform(sub_data) 
        
        df = pd.DataFrame(scaled_data, columns=self.cols_list)
        final_data = pd.concat([other_data, df], ignore_index=True, axis=1)
        final_data.columns = other_cols + self.cols_list
        # print(final_data.head()); sys.exit()
        return final_data
    
    
    def inverse_transform(self, data): 
        if len(data.shape) == 1:
            data = data.reshape((-1,1))
        rescaled_data = self.scaler.inverse_transform(data) 
        
        # rescaled_data = pd.DataFrame(rescaled_data, columns=self.cols_list)
        return rescaled_data



class CustomStandardScaler(CustomScaler): 
    def __init__(self, cols_list):
        super(CustomStandardScaler, self).__init__(cols_list)
        self.scaler = StandardScaler()
        


class CustomYeoJohnsonTransformer(CustomScaler): 
    def __init__(self, cols_list):
        super(CustomYeoJohnsonTransformer, self).__init__(cols_list)
        self.scaler = PowerTransformer(method="yeo-johnson", standardize=False)
        # self.scaler = PowerTransformer(method="box-cox")
        
        
class CustomMinMaxScaler(CustomScaler): 
    def __init__(self, cols_list):
        super(CustomMinMaxScaler, self).__init__(cols_list)
        self.scaler = MinMaxScaler()


class CustomQuintileTransformer(CustomScaler): 
    def __init__(self, cols_list):
        super(CustomQuintileTransformer, self).__init__(cols_list)
        self.scaler = QuantileTransformer(n_quantiles=10000)
        
        
class MinMaxBounder(BaseEstimator, TransformerMixin): 
    def __init__(self, cols_list):
        self.cols_list = cols_list
        
        
    def fit(self, X, y=None): 
        sub_df = X[self.cols_list]
        self.min_vals = dict(sub_df.min())
        self.max_vals = dict(sub_df.max())
        return self
    
    
    def transform(self, data):
        
        data.reset_index(inplace=True, drop=True)
        
        other_cols = [col for col in data.columns if col not in self.cols_list]
        other_data = data[other_cols]
        
        sub_data = data[self.cols_list]
        
        bounded_data = sub_data.clip(
            lower=pd.Series(self.min_vals), 
            upper=pd.Series(self.max_vals), axis=1)
        
        
        df = pd.DataFrame(bounded_data, columns=self.cols_list)
        final_data = pd.concat([other_data, df], ignore_index=True, axis=1)
        final_data.columns = other_cols + self.cols_list        
        
        return final_data
    
    
    def inverse_transform(self, data): 
        
        df = pd.DataFrame(data, columns=self.cols_list)
        
        bounded_data = df.clip(
            lower=pd.Series(self.min_vals), 
            upper=pd.Series(self.max_vals), axis=1)
   
        return bounded_data.values
    


class TargetFeatureAdder(BaseEstimator, TransformerMixin): 
    def __init__(self, label_field_name) -> None:
        super().__init__()
        self.label_field_name = label_field_name
    
    def fit(self, data): return self
    
    def transform(self, data): 
        if self.label_field_name not in data.columns: 
            data[self.label_field_name] = 0.
        return data


class ValueClipper(BaseEstimator, TransformerMixin): 
    def __init__(self, fields_to_clip, min_val, max_val) -> None:
        super().__init__()
        self.fields_to_clip = fields_to_clip
        self.min_val = min_val
        self.max_val = max_val
    
    def fit(self, data): return self
    
    def transform(self, data): 
        for field in self.fields_to_clip:
            if self.min_val is not None: 
                data[field] = data[field].clip(lower=self.min_val)
            if self.max_val is not None: 
                data[field] = data[field].clip(upper=self.max_val)
        return data


class XYSplitter(BaseEstimator, TransformerMixin): 
    def __init__(self, target_col, id_col):
        self.target_col = target_col
        self.id_col = id_col
    
    def fit(self, data): return self
    
    def transform(self, data):
        # data.to_csv("data.csv", index=False); sys.exit()
        if self.target_col in data.columns: 
            y = data[self.target_col].values
        else: 
            y = None
        
        not_X_cols = [ self.id_col, self.target_col ] 
        X_cols = [ col for col in data.columns if col not in not_X_cols ]
        X = data[X_cols]
        
        return { 'X': X, 'y': y, 'ids': data[self.id_col].values }
    
        
    
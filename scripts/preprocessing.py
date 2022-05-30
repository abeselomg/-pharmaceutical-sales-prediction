from cProfile import label
import pandas as pd
import numpy as np
from sklearn import preprocessing
from script_logger import App_Logger

class Preprocessor:
    def __init__(self,df):
        self.df = df
        self.logger = App_Logger("script.log").get_app_logger()
        
        
    def type_to_str(self,col):
        self.df[col] = self.df[col].astype(str)
        return self.df
    
    
    def add_weekend_col(self):
          
        self.df["Weekends"] = self.df["DayOfWeek"].apply(lambda x: 1 if x > 5 else 0)
        self.df["Weekends"] = self.df["Weekends"].astype("category")
        self.logger.info("added weekend column")
        return self.df
        
        
        
    def transform_date(self ):
            
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['Year'] = pd.DatetimeIndex(self.df['Date']).year.astype("category")
        self.df['Month'] = pd.DatetimeIndex(self.df['Date']).month.astype("category")
        self.df['Day'] = pd.DatetimeIndex(self.df['Date']).day.astype("category")
        self.logger.info("Date separated into year, month and day")
        return self.df
        
        
    def handle_null_objects(self):
        try:
            obj_null=self.df.select_dtypes(include=["object"]).isna().any().keys().tolist()
            for obj in obj_null:
                self.df[obj].fillna(self.df[obj].mode().iloc[0],inplace=True)
            
            self.logger.info("Filled null values with object datatypes")
            
            
        except Exception as e:
            self.logger.error("Error while trying to fill null values with object datatypes")

    
    def handle_null_numbers(self):

        mean_df,median_df=self.handle_skewness()
        if mean_df is not None:
            for col in mean_df.columns.to_list():
                self.df[col].fillna(self.df[col].mean(), inplace=True)
                
            for col in median_df.columns.to_list():
                self.df[col].fillna(self.df[col].median(), inplace=True)
                
            self.logger.info("Filled null values with numeric datatypes")
            
        
        
    def handle_skewness(self):
        try:
            #get columns with null values of numeric type
            num_null = self.df.select_dtypes(include=["number"]).notna().any().keys().tolist()
            #calculate skewness
            self.df[num_null].skew().sort_values(ascending=False)
            name = self.df[num_null].skew().keys().tolist()
            
            skew_df = pd.DataFrame([self.df[num_null].skew().to_numpy()], columns=name)
            
            # change the values of skewed columns for mean or median
            for_mean_df = skew_df[(skew_df > 1) | (skew_df < -1)]
            for_median_df = skew_df[(skew_df < 1) & (skew_df > -1)]
            
            self.logger.info("Skewness calculated for numeric columns and changed values for mean or median")

            for_mean_df.dropna(inplace=True, axis=1)
            for_median_df.dropna(inplace=True, axis=1)
            return for_mean_df,for_median_df
        except Exception as e:
            self.logger.error("Error while trying to fill null values with numeric datatypes")
            
            
            
    def handle_outliers(self, df, col):
        
        df = df.copy()
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        lower_bound = q1 - ((1.5) * (q3 - q1))
        upper_bound = q3 + ((1.5) * (q3 - q1))
        
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

        return df
    
    

        
  
            
    def return_clean_data(self):
        self.df=self.type_to_str("StateHoliday")
        self.df = self.transform_date()
        self.df = self.add_weekend_col()
        self.handle_null_objects()
        self.handle_null_numbers()
        return self.df
           
        


    
    

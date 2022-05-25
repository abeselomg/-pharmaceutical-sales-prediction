import pandas as pd

class DataExploration:
    def __init__(self, df):

        self.df = df

    def read_head(self, top=5):
        return self.df.head(top)

    # returning the number of rows columns and column information
    def get_info(self):
        row_count, col_count = self.df.shape

        print(f"Number of rows: {row_count}")
        print(f"Number of columns: {col_count}")
        print("================================")

        return (row_count, col_count), self.df.info()

    def get_count(self, column_name):
        return pd.DataFrame(self.df[column_name].value_counts())

    # getting the null count for every column
    def get_null_count(self, column_name):
        print("Null values count")
        print(self.df.isnull().sum())
        return self.df.isnull().sum()

    # getting the percentage of missing values
    def get_null_percentage_of_dataframe(self) -> pd.DataFrame:
        return self.df.isnull().sum().sort_values(ascending=False) / self.df.shape[0] * 100
    
    def transform_date(self, df):
        
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = pd.DatetimeIndex(df['Date']).year
        df['Month'] = pd.DatetimeIndex(df['Date']).month
        df['Day'] = pd.DatetimeIndex(df['Date']).day
        return df

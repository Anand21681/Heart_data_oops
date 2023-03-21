import pickle
import pandas as pd 
import numpy as np 
import CONFIG

class prediction:
    
    def data(self):
        with open(CONFIG.model_path,'rb')as file:
            self.std_model=pickle.load(file)

        with open(CONFIG.scaler_path,'rb')as file:
            self.std_scaler=pickle.load(file)

        self.std_model
        self.std_scaler
    
    def class_predict(self,input_data):
        self.data()
        S1=pd.Series(input_data,index=input_data.keys())
        df = pd.DataFrame(S1)
        trans_df=df.transpose()
        x_df=trans_df

        arr=self.std_scaler.transform(x_df)
        std_df=pd.DataFrame(arr,columns=trans_df.columns)

        pred=self.std_model.predict(std_df)
        return pred

        

if __name__=="__main__":

    pred_obj=prediction()
    pred_obj.class_predict(input_data)

        




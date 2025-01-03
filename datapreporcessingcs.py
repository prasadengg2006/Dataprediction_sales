import numpy as np
import streamlit as st
import pandas as pd

import seaborn as sns
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from prophet import Prophet

import warnings
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA



pizza_df=pd.read_excel("F:\datasets\Dominos_dataset\Pizza_Sale.xlsx")
ingredients_df=pd.read_excel("F:\datasets\Dominos_dataset\Pizza_ingredients.xlsx")
#df.fillna(method='ffill', inplace=True)


info=pizza_df.info()
st.write(info)
st.write(pizza_df.isna().sum())
pizza_df.dropna(inplace=True)
st.write(pizza_df.isna().sum())
duplicates=pizza_df.duplicated().sum()
st.write(duplicates)

def convert_ToDate(date):
    for fmt in('%d-%m-%Y','%d/%m/%Y'):
        try:
            return pd.to_datetime(date,format=fmt)
        except ValueError:
            pass
    raise ValueError(f'no validate format found for {date}')   

pizza_df['order_date']=pizza_df['order_date'].apply(convert_ToDate)  
orderdate=pizza_df['order_date'].head()  
st.write(pizza_df) 
st.write(pizza_df.shape)
Qty_period=pizza_df.groupby('order_date')['quantity'].sum()
fig=plt.figure(figsize=(20,12))
plt.plot(Qty_period.index,Qty_period.values)
plt.xlabel('Order Date')
plt.ylabel('Qunatity Sold')
plt.title('Quantity Sold Over Time')
plt.grid(True)
#plt.xticks(rotation=90, ha='right')

st.pyplot(fig)
figcorrelation=plt.figure(figsize=(12,8))
ax1=figcorrelation.add_subplot(211)
figcorrelation=plot_acf(Qty_period,lags=40,ax=ax1)
ax2=figcorrelation.add_subplot(212)
figcorrelation=plot_pacf(Qty_period, lags=40, ax=ax2)

st.pyplot(figcorrelation)


def adfuller_test(sales):
    result=adfuller(sales)
    st.write('adfuller statistics: %f' % result[0])
    st.write('p-value : %f' % result[1])
    st.write(f'# logsused= {result[2]}')
    st.write(f'# observationsused= {result[3]}')
    if result[1]<=0.05:
        st.write ('we reject the null hypothesis, the series is stationary')
    else :
        st.write('Not enough statistical evidence to reject null hypothesis')

adfuller_test(Qty_period)

evaluation_data=pizza_df.groupby(['order_date','pizza_name'])['quantity'].sum().unstack().fillna(0)

st.write(evaluation_data.head())

one_pizza= evaluation_data['The Italian Supreme Pizza']

#train and test
train=one_pizza[:-7]
test=one_pizza[-7:]

st.write(train)
st.write(test)

ari_model=ARIMA(train,order=(1,1,0))
ari_result=ari_model.fit()

ari_forecast=ari_result.get_forecast(steps=len(test))
ari_fore_values=ari_forecast.predicted_mean
ari_predict=ari_result.predict(start=len(train),end=len(train)+len(test)-1, dynamic=True)
ari_predict.index=test.index

concatenated_series=pd.concat([test,ari_predict],axis=1) 
concatenated_series.columns=['Actual','Predicted']
#st.write(concatenated_series)
#cs=concatenated_series.plot(title="Actual vs Predicted")
#concatenated_series.plot(figsize=(12,6))
#fig_actual_predict=plt.figure(figsize=(12,6))

plt.xlabel('ordered date')
plt.xlabel('quantity sold')
plt.title('Actual vs predicted')
fig=px.line(concatenated_series)

#fig=px.line(concatenated_series,x='Predicted',y='Actual',labels={'y':'Actual'})
#fig.add_scatter(x=concatenated_series['Actual'],y=concatenated_series['Predicted'], mode='lines',name='Predicted')
fig.update_layout(title='Actual Vs Predicted', xaxis_title='Date', yaxis_title='Value')
st.plotly_chart(fig,use_container_width=True)


#Evaluate the mode
mae=mean_absolute_error(test,ari_fore_values)
mse=mean_squared_error(test,ari_fore_values)
rmse=np.sqrt(mse)
st.write(f'mae :{mae}')
st.write(f'mse :{mse}')
st.write(f'rmse :{rmse}')


sari_model=SARIMAX(train,order=(1,1,1),seasonal_order=(1,1,1,7))
sari_result=sari_model.fit()

sari_forecast=sari_result.get_forecast(steps=len(test))
sari_fore_values=sari_forecast.predicted_mean

mae=mean_absolute_error(test,sari_fore_values)
mse=mean_squared_error(test,sari_fore_values)
rmse=np.sqrt(mse)
st.write(f' sari mae :{mae}')
st.write(f'sari mse :{mse}')
st.write(f'sari rmse :{rmse}')

sari_predict=sari_result.predict(start=len(train),end=len(train)+len(test)-1,dynamic=True)
sari_predict.index=test.index
st.write(sari_predict)

concatenated_sarima=pd.concat([test,sari_predict],axis=1)
concatenated_sarima.columns=['Actual','Predicted']
st.write(concatenated_sarima)
fig=px.line(concatenated_sarima)
#fig.add_scatter(x=concatenated_sarima['order_date'],y=concatenated_sarima['Predicted'], mode='lines',name='Predicted')
fig.update_layout(title='Actual Vs Predicted', xaxis_title='Date', yaxis_title='Value')
st.plotly_chart(fig,use_container_width=True)


one_pizza_prophet=pd.DataFrame({'ds':one_pizza.index,'y':one_pizza.values})
st.write(one_pizza_prophet.head())

pr_train=one_pizza_prophet[:-7]
pr_test=one_pizza_prophet[-7:]

st.write(pr_train)
st.write(pr_test)
pr_model=Prophet()
pr_model.fit(pr_train)

pr_future=pr_model.make_future_dataframe(periods=7,freq='D')
pr_forecast=pr_model.predict(pr_future)
pr_forecast_values=pr_forecast['yhat'][-len(pr_test):].values
mae=mean_absolute_error(pr_test['y'],pr_forecast_values)
mse=mean_squared_error(pr_test['y'],pr_forecast_values)
rmse=np.sqrt(mse)
st.write(f' prophet mae :{mae}')
st.write(f'prophet mse :{mse}')
st.write(f'prophet rmse :{rmse}')

concat_prophet=pd.concat([pr_test,pr_forecast[['yhat']].iloc[-len(pr_test):]],axis=1)
concat_prophet.columns=['date','Actual','Predicted']
#concat_prophet.set_index('date',inplace=True)
st.write(concat_prophet)

fig=px.line(concat_prophet,x='date',y='Actual',labels={'y':'Actual'})
fig.add_scatter(x=concat_prophet['date'],y=concat_prophet['Predicted'], mode='lines',name='Predicted')
fig.update_layout(title='Actual Vs Predicted', xaxis_title='Date', yaxis_title='Value')
st.plotly_chart(fig,use_container_width=True)

pizza_df=pizza_df[['order_date','pizza_name','quantity']]
sales_summary=pizza_df.groupby(['order_date','pizza_name'])['quantity'].sum().reset_index()
st.write(sales_summary.head())

sales_pivot=sales_summary.pivot(index='order_date',columns='pizza_name',values='quantity').fillna(0)
st.write(sales_pivot)
arima_models={}
for pizza_name in sales_pivot.columns:
    try:
        model=ARIMA(sales_pivot[pizza_name],order=(1,1,0))
        model_fit=model.fit()
        arima_models[pizza_name]=model_fit
    except:
        print(f'Arima model for {pizza_name} failed for fit')

#generate predictions for one week 
prediction_days=7
predictions_arima={}
st.write(arima_models)
for pizza_name,model in arima_models.items():
    predictions_arima[pizza_name]=model.predict(start=len(sales_pivot),end=len(sales_pivot)+ prediction_days-1)

predictions_df=pd.DataFrame(predictions_arima)
predictions_df.index=pd.date_range(start=sales_pivot.index[-1],periods=prediction_days,freq='D')

ingred_df=ingredients_df[['pizza_name','pizza_ingredients','Items_Qty_In_Grams']]
ingred_df.rename(columns={'Items_Qty_In_Grams':'items_qty'},inplace=True)

st.write(predictions_df)
st.write(ingred_df.head())

ingridient_quantities={}

for pizza_name in predictions_df.columns:
    predicted_qty=predictions_df[pizza_name].sum()
    pizza_ingredients=ingred_df[ingred_df['pizza_name']==pizza_name]  
    for index, row in pizza_ingredients.iterrows():
     ingredient=row['pizza_ingredients']
     ingredient_qty=row['items_qty']
     required_qty=predicted_qty*ingredient_qty

     if ingredient not in ingridient_quantities:
         ingridient_quantities[ingredient]=0
     ingridient_quantities[ingredient]=required_qty  


ingridient_requirements_df=pd.DataFrame.from_dict(ingridient_quantities,orient='index',columns=['required_quantity']) 
st.write(ingridient_requirements_df)    


purchase_order_df=ingridient_requirements_df.copy()
purchase_order_df['units']='grams'
purchase_order_df=purchase_order_df.rename(columns={'required_Quantity':'quantity'})
st.write(purchase_order_df)

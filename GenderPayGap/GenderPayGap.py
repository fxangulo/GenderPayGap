'''
    GenderPayGap analysis 2022
DESCRIPTION
    This module provides methods to automate the Gender Pay Gap analysis. 
    It performs the exploratory data analysis and data modeling to obtain
    the Salary Gap Decomposition, the Adjusted Gender Pay Gap and 
    the significant variable coefficients. 
MODULE CONTENTS
    CLASS
    - GenderPayGap(df, bifurcate, salary, df_dummy=None, df_significant=None, swap=False)
        METHODS
        - exploratory_data_analysis(self, polyn=2)
        - poly_plot(self, polyn=2)
        - prepare_data (self, max_unique=45, column_to_exp='', exponent=2, 
           column_to_log='', log_function='log2', drop_original=True)
        - select_significant (self, p_limit=0.05, to_drop='')
        - plot_coefficients (self)
        - avg_decomposition(self, width=None, height=None)
        - gap_decomposition(self, width=None, height=None)
        - gap_summary(self)
        - correlation_matrix (self, df=None)
'''

# Libraries to import
import pandas as pd
import numpy as np

# libraries for data visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# To show Plotly figures when downloading the notebook in .html
import plotly.io as pio
pio.renderers.default='notebook' 

# libraries for building linear regression model
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# Display formatted tables from functions
from IPython.display import display

# Remove scientific notations and display numbers with 2 decimal points instead
pd.options.display.float_format = '{:,.2f}'.format      

# To ignore warnings
import warnings
warnings.filterwarnings("ignore")


class GenderPayGap ():
    
    def __init__(self, df, bifurcate, salary, df_dummy=None, df_significant=None, swap=False):
        '''
        Calculates average and raw gap between two groups.
        Requires a dataframe with bufurcate and values column names.
        Returns p_gap dataframe
        '''
        # Validate the bifurcate column has only 2 values
        if df[bifurcate].nunique()!=2:
            print("Bifurcate column must have 2 values to be executed")
        # Select reference and to-explain groups 
        else: 
            self.pay_gap =pd.DataFrame(df.groupby(bifurcate)[salary].mean()).T
            if swap != False:
                # Rearrange reference and to explain columns
                cols= self.pay_gap.columns.tolist()[::-1]
                self.pay_gap=self.pay_gap[cols]
            
            self.reference= self.pay_gap.columns[0] 
            self.to_explain= self.pay_gap.columns[1]
            self.pay_gap['RawGAP'] = self.pay_gap.iloc[:,0].sub(self.pay_gap.iloc[:,1], axis=0)
            self.pay_gap['%RawGAP']= self.pay_gap['RawGAP'].div(self.pay_gap.iloc[:,0])*100

            self.df=df # The dataframe with all variables
            self.bifurcate=bifurcate # A string with the name of the column
            self.salary=salary # A string with the name of the dependent variable
            self.df_dummy= None 
            self.df_significant=df_significant # A dataframe with only significant and encoded variables
            self.ols_first=None
            self.ols_final=None
            self.bifurcate_encoded= None
            self.p_limit= None
            self.df_ols=None 
            self.df_gap_decomposition=None
            self.ols_gap_decomposition=None
            display(self.pay_gap)

            
    def exploratory_data_analysis(self, polyn=2):
        '''
        Univariable exploratory data analysis. It also calls poly-plot method
        '''
        # Plot numerical variables
        for col in self.df.select_dtypes('number').columns:
            print(col)
            print('Skew :',round(self.df[col].skew(),2))
            display(self.df.groupby(self.bifurcate)[col].describe())
            plt.figure(figsize=(15,4))
            plt.subplot(1,2,1)
            sns.histplot(x=col, data=self.df, hue=self.bifurcate, multiple="stack")
            plt.ylabel('count')
            plt.subplot(1,2,2)
            sns.boxplot(x=col, y=self.bifurcate, data=self.df) 
            plt.show()
        # Plot object variables
        for col in self.df.select_dtypes('object').columns:
            display(pd.DataFrame(self.df[col].describe(include=object)).T)# Describe categorical variables
            unique=self.df[col].nunique()*0.6
            if unique>12:
                unique=12
            plt.figure(figsize=(14, unique))
            sns.countplot(y=col, data=self.df.sort_values(by=[col]), hue=self.bifurcate) #sort_values(by=[col])
            plt.show()
        self.poly_plot(polyn)

        
    def poly_plot(self, polyn=2): 
        #Creates scatter plots with lineal an polynomial regression for number variables and salary
        print('Show polinomial plots for numerical variables')
        # Selects dependent and independent numerical variables
        df_poly=self.df.copy()
        y_col=self.salary
        y1= df_poly[y_col].values.copy()
        x_cols= df_poly.select_dtypes('number').columns.tolist()
        x_cols.remove(y_col)
        poly_list=[1]
        poly_list.append(polyn)
        # Plots variables
        for x_col in x_cols:
            x1= df_poly[x_col].values.copy()
            position=1
            plt.figure(figsize=(15,4))
            for i in poly_list:
                #Plot original value
                plt.subplot(1,2,position)
                sns.scatterplot(data=df_poly,x=x_col,y=y_col, hue=self.bifurcate)
                #calculate equation for polynominal trendline
                z = np.polyfit(x1,y1, i)
                poly = np.poly1d(z)
                # Calculate new x and y values
                new_x = np.linspace(x1.min(), x1.max()) # Create the points for the curve
                new_y = poly(new_x)
                #add trendline to plot 
                plt.plot(new_x, new_y, 'r')
                df_poly['poly']=df_poly[x_col]**i
                x0= df_poly['poly'].values.reshape(len(df_poly[x_col]),1)
                #Print linear regression coefficient
                LR1= LinearRegression()
                LR2= LR1.fit(x0,y1)
                plt.title('Poly= '+str(i)+'  |  R-square= '+str(LR2.score(x0,y1)))
                position=position+1
        return        

    
    def prepare_data (self, max_unique=45, column_to_exp='', exponent=2, column_to_log='', log_function='log2', drop_original=True):
        '''
        One hot encoding for categorical data and data modeling-
        exponentiate or logarithm for the specified columns
        '''
        # Check if there are variables to encode
        cat_cols= self.df.select_dtypes(['object']).columns.tolist()
        if len(cat_cols)>0:      
            print('Identified columns to encode... ', list(cat_cols))
            df_to_encode=self.df.copy()
            #Check the bifurcate column to encode
            if not (np.issubdtype(df_to_encode[self.bifurcate].dtypes, np.number)):
                #Encoding bifurcate column as required in init swap and saves new bifurcate encoded name
                encode_format = {self.reference: 0, self.to_explain: 1}
                bifurcate_encoded= self.bifurcate + '_'+self.to_explain
                df_to_encode[self.bifurcate] = df_to_encode[self.bifurcate].map(encode_format)
                df_to_encode.rename(columns={ self.bifurcate: bifurcate_encoded }, inplace=True)
                self.bifurcate_encoded=bifurcate_encoded
            else:
                self.bifurcate_encoded= self.bifurcate
            #Convert object columns into dummy variables
            cat_cols= df_to_encode.select_dtypes(['object']).columns.tolist()
            #Checks if there are more columns for dummification
            if len(cat_cols)>0:      
                to_encode=[]
                for i in cat_cols:
                    count= df_to_encode[i].nunique()
                    if count >max_unique:      #Excludes columns for dummification when exceeds max unique values
                        print('Excluded for encoding and drop... ', i, count, '(more than ', max_unique, ' values)')
                        df_to_encode.drop([i], axis=1, inplace=True)
                    else:
                        to_encode.append(i)
                        print('Included for encoding.......... ',i)
                dummy_df= pd.get_dummies(data=df_to_encode, columns=to_encode, drop_first= True)
            else:
                dummy_df= df_to_encode    
        else:
            dummy_df= self.df
            print('No object columns identified to enconde')

        # Exponentiate the required column or list of columns 
        if column_to_exp !='':
            if type(column_to_exp) is list:
                for j in column_to_exp:
                    new_name= j+'**'+str(exponent)
                    dummy_df[new_name]= dummy_df[j]**exponent
                    print('New column added............... ' ,new_name, '(', j, 'raised to the power of', exponent,')')
                    if drop_original == True:
                        dummy_df.drop(columns={j}, inplace=True)
                        print('Original column droped........... ' ,j)
                    else:
                        print('Original column keeped........... ' ,j)
            elif type(column_to_exp) is str:
                new_name= column_to_exp+'**'+str(exponent)
                dummy_df[new_name]= dummy_df[column_to_exp]**exponent
                print('New column added............... ' ,new_name, '(', column_to_exp, 'raised to the power of', exponent,')')
                if drop_original == True:
                    dummy_df.drop(columns={column_to_exp}, inplace=True)
                    print('Original column droped........... ' ,column_to_exp)
                else:
                    print('Original column keeped........... ' ,column_to_exp)
            else:
                print('Column(s) for exponentiation must be str or list')

        # Apply the specified log to the required column or list of columns
        if column_to_log !='':
            if type(column_to_log) is list:
                for k in column_to_log:
                    new_name= k+'_'+log_function
                    if log_function=='log2':
                        dummy_df[new_name]= np.log2(dummy_df[k])
                    elif log_function=='log':
                        dummy_df[new_name]= np.log(dummy_df[k])
                    elif log_function=='log10':
                        dummy_df[new_name]= np.log10(dummy_df[k])
                    else:
                        print('log_function must be "log2" (default), "log" or "log10"')
                    print('New column added............... ' ,new_name, '(', log_function, 'of', k,')')
                    if drop_original == True:
                        dummy_df.drop(columns={k}, inplace=True)
                        print('Original column droped........... ' ,k)
                    else:
                        print('Original column keeped........... ' ,k)
            elif type(column_to_log) is str:
                new_name= column_to_log+'_'+log_function
                if log_function=='log2':
                    dummy_df[new_name]= np.log2(dummy_df[col_log])
                elif log_function=='log':
                    dummy_df[new_name]= np.log(dummy_df[col_log])
                elif log_function=='log10':
                    dummy_df[new_name]= np.log10(dummy_df[col_log])
                else:
                    print('log_function must be "log2" (default), "log" or "log10"')
                print('New column added............... ' ,new_name, '(', log_function, 'of', column_to_log, ')')
                if drop_original == True:
                    dummy_df.drop(columns={column_to_log}, inplace=True)
                    print('Original column droped........... ' ,column_to_log)
                else:
                    print('Original column keeped........... ' ,column_to_log)
            else:
                print('Column(s) for log must be str or list')
        # Closing the method
        print('New dataframe total columns.... ', len(dummy_df.columns))
        self.df_dummy= dummy_df
        display(self.df_dummy.head().T)


    def select_significant (self, p_limit=0.05, to_drop=''):
        '''
        Performs statsmodels OLS regression and selects p-value significant columns
        '''
        self.p_limit=p_limit
        # Checks that df_dummy has already been created
        if self.df_dummy is None:
            print('Method select_significant can not be processed')
            print ('...dataframe is not dummified') 
            print('...please call df_prepare first')
        else:
            self.df_dummy_new=self.df_dummy.copy()
            print('Initial columns.................................... ', len(self.df_dummy_new.columns))

            # If required, it drops the specified column/s
            if to_drop !='':
                self.df_dummy_new= self.df_dummy_new.drop(columns=to_drop)
                if type(to_drop) is str:
                    print('Columns to drop.................................... ', to_drop)
                elif type(to_drop) is list:
                    print('Columns to drop.................................... ', to_drop)
                else: 
                    print('Columns to drop must be string or list')

            # Performs the statsmodel OLS method adding constant
            self.df_dummy_new =sm.add_constant(self.df_dummy_new) 
            print('Constant column added for Ordinary Least Squares regression')
            #create and fit the model
            self.ols_first = sm.OLS(self.df_dummy_new[self.salary], self.df_dummy_new.drop([self.salary], axis=1)).fit()  
            print ('Adjusted r-square with original variables ......... ', self.ols_first.rsquared_adj)
            # Creates a dataframe with p-values
            p_values= pd.DataFrame(self.ols_first.pvalues).reset_index().rename(columns = {'index':'variable', 0:'p-value'})
            # Selects columns with p value > p_limit
            cols_del= p_values.variable[p_values['p-value']>p_limit].tolist()
            if self.bifurcate_encoded in cols_del:
                cols_del.remove(self.bifurcate_encoded)
                print ('IMPORTANT: p-value of ', self.bifurcate_encoded, 'is not significant: ', 
                       self.ols_first.pvalues.loc[self.bifurcate_encoded] )
                print (self.bifurcate_encoded, 'will not be removed')
            self.df_significant = self.df_dummy_new.drop(columns=cols_del)
            print('Variables to drop ( "p-value" >', p_limit,')............. ', len(cols_del))
            print('Variables dropped:................................. ', cols_del)
            #create and fit the model with significant variables
            self.ols_final = sm.OLS(self.df_significant[self.salary], self.df_significant.drop([self.salary], axis=1)).fit()  
            print('Adjuster r-square with significant variables....... ', self.ols_final.rsquared_adj)
            print('Final variables considered......................... ', len(self.df_significant.columns))
            display(self.df_significant.head().T)
        

    def plot_coefficients (self):
        '''
        Get the OLS coefficients and plots them
        '''
        # Obtain the list with coefficients
        coef = pd.DataFrame(self.ols_final.params,columns=['Coefficients'])
        coef = coef.sort_values(by='Coefficients' , axis=0, ascending=False, inplace=False)
        graph=px.bar( y=coef.Coefficients , x=coef.index,              
                title='Coefficients',  text = round(coef.Coefficients,2),
                orientation='v', width=800, height=600 , color=coef.Coefficients, 
                labels={'y':'Variables', 'x': 'Coefficient'})
        return graph
    
        
    def avg_decomposition(self, width=None, height=None):
        '''
        Calculates and creates a table and plot with basic statistics 
        (mean, max, min and standard deviation), the coefficients and 
        the salary average related to each independent variable
        '''
        # If no OLS instances passed, fits the df_dummy dataframe
        if self.ols_final != None:
            ols_instance = self.ols_final
            df_ols= self.df_significant.copy()
            print('Salary decomposition with significant p-values (>',self.p_limit,')' )
        elif isinstance(self.df_dummy, pd.DataFrame):
            df_ols=self.df_dummy.copy()
            df_ols=sm.add_constant(df_ols) # add constant
            ols_instance= sm.OLS(df_ols[self.salary], df_ols.drop([salary], axis=1)).fit() 
            print('IMPORTANT: select_significant method has not been executed')
            print('OLS instance wll be done with dummified dataframe and will not select significant p-values')
        else:
            print('Please execute df_prepare and select_signifcant methods first')
            return None
            
       # Obtains the list with coefficients
        coef = pd.DataFrame(ols_instance.params,columns=['Coefficients'])
       # Calculate std, mean, min & max and takes them into a df with the coeficients
        df_decomp= pd.concat([df_ols.drop([self.salary], axis=1).std(axis=0), 
                             df_ols.drop([self.salary], axis=1).min(axis=0), 
                             df_ols.drop([self.salary], axis=1).max(axis=0),
                             df_ols.drop([self.salary], axis=1).mean(axis=0),
                             coef], axis=1, join="inner") 
        df_decomp.rename(columns={0:'Value_STD', 1:'Value_MIN', 2:'Value_MAX', 3:'Value_MEAN'}, inplace=True)

        # Calculates Salary_STD(Value_STD * coefficients) and Salary_MEAN (Value_MEAN * coefficients)
        df_decomp['Salary_STD']= df_decomp['Value_STD'].mul(df_decomp['Coefficients'], axis = 0)
        df_decomp['Salary_MEAN']= df_decomp['Value_MEAN'].mul(df_decomp['Coefficients'], axis = 0)

        # Sorts variables by Salary_MEAN and introduces Total sum
        df_decomp= df_decomp.sort_values(by='Salary_MEAN' , axis=0, ascending=False, inplace=False)
        df_decomp.loc['Total'] = df_decomp.sum()

        # Configure the measure parameter to construct the waterfall graph
        measures=[]
        for n in range(len(df_decomp)-1):
            var = "relative" +','
            measures.append(var)
        measures.append('total')
       # Plot salary average salary decomposition
        fig_decomp = go.Figure(go.Waterfall(
        name = "Contribution", orientation = "v",
        measure = measures,
        x = df_decomp.index,
        textposition = "outside",
        text = round(df_decomp['Salary_MEAN'],0),
        y = df_decomp['Salary_MEAN'], connector_visible=False))
        fig_decomp.update_layout(title = "Average salary decomposition",showlegend = True)
        fig_decomp.update_layout(margin=dict( l=50, r=50,b=100,t=40,pad=4))
        if width!=None and height!=None:
            w=int(width)
            h=int(height)
            print('Specified width and height')
            fig_decomp.update_layout(autosize=False, width=w, height=h)
        elif width!=None or height!=None:
            print('Please enter valid width and height')
        else:
            print('No width and height specified')
        self.df_avg_salary_decomp= df_decomp
        self.plot_avg_salary_decomp= fig_decomp
        fig_decomp.show()
        return self.df_avg_salary_decomp

    
    def gap_decomposition(self, width=None, height=None):
        '''
        Calculates the average values of men and women, the value’s gap, the coefficients, 
        the salary average per gender and the gap corresponding to each independent variable
        '''    
        #If no OLS instances passed, fits the dataframe
        
        if self.ols_final != None:
            ols_instance = self.ols_final
            df_ols= self.df_significant.copy()
            print('Salary decomposition with significant p-values (>',self.p_limit,')' )
        elif isinstance(self.df_dummy, pd.DataFrame):
            df_ols=self.df_dummy.copy()
            df_ols=sm.add_constant(df_ols) # add constant
            ols_instance= sm.OLS(df_ols[self.salary], df_ols.drop([salary], axis=1)).fit() 
            print('IMPORTANT: select_significant method has not been executed')
            print('OLS instance wll be done with dummified dataframe and will not select significant p-values')
        else:
            print('Please execute df_prepare and select_signifcant methods first')
            return None
        
        #Obtains the list with coefficients
        coef = pd.DataFrame(ols_instance.params,columns=['Coefficients'])
        #Obtain averages per group and include the bifurcate row to join with df_decomp
        df_decomp_bi= df_ols.drop([self.salary], axis=1).groupby(self.bifurcate_encoded).mean().T
        df_decomp_bi.rename(columns={0:self.reference, 1:self.to_explain}, inplace=True) 
        df_decomp_bi['Value_GAP']= df_decomp_bi[self.reference].sub(df_decomp_bi[self.to_explain], axis = 0)
        df_decomp_bi= pd.concat([df_decomp_bi, coef], axis=1, join="outer") 
        df_decomp_bi[self.to_explain].loc[self.bifurcate_encoded]=1 #replace with bifurcate
        df_decomp_bi[self.reference].loc[self.bifurcate_encoded]=0 #replace with bifurcate

        #Join tables and obtain average salary per values
        self.reference_head= self.reference+'_PAY'
        self.to_explain_head= self.to_explain+'_PAY'
        df_decomp_bi[self.reference_head]= df_decomp_bi[self.reference].mul(df_decomp_bi['Coefficients'], axis = 0)
        df_decomp_bi[self.to_explain_head]= df_decomp_bi[self.to_explain].mul(df_decomp_bi['Coefficients'], axis = 0)
        df_decomp_bi['Salary_GAP']= df_decomp_bi[self.reference_head].sub(df_decomp_bi[self.to_explain_head], axis = 0)
                
        # Sorts variables by Salary_MEAN and introduces Total sum
        df_decomp_bi= df_decomp_bi.sort_values(by='Salary_GAP' , axis=0, ascending=False, inplace=False)
        df_decomp_bi.loc['Total'] = df_decomp_bi.sum() 
        df_decomp_bi= round(df_decomp_bi,2)
        df_decomp_bi['Percentage_GAP']= df_decomp_bi['Salary_GAP'].div(df_decomp_bi[self.reference_head], axis=0)*100

        # Configure the measure parameter to construct the waterfall graph
        measures=[]
        for n in range(len(df_decomp_bi)-1):
            var = "relative" +','
            measures.append(var)
        measures.append('total')
        #Plot salary GAP decomposition
        fig_gpg = go.Figure(go.Waterfall(
        name = "Adjusted GAP decomposition", orientation = "v",
        measure = measures,
        x = df_decomp_bi.index,
        textposition = "outside",
        text = round(df_decomp_bi['Salary_GAP'],0),
        y = df_decomp_bi['Salary_GAP'], connector_visible=False))
        fig_gpg.update_layout(title = "Adjusted Gender Pay Gap decomposition",showlegend = True)
        fig_gpg.update_layout(margin=dict( l=50, r=50,b=100,t=40,pad=4))
        if width!=None and height!=None:
            w=int(width)
            h=int(height)
            print('Specified width and height')
            fig_gpg.update_layout(autosize=False, width=w, height=h)
        elif width!=None or height!=None:
            print('Please enter valid width and height')
        else:
            print('No width and height specified')
        self.ols_gap_decomposition= ols_instance
        self.df_gap_decomposition= df_decomp_bi
        self.plot_gap_salary_decomp= fig_gpg
        fig_gpg.show()
        return self.df_gap_decomposition

    
    def gap_summary(self):
        '''
        Calculates the adjusted pay gap and calculates Oaxaca-Blinder two fold
        '''
        if self.ols_gap_decomposition is None:
            print('Please execute gap_salary_decomp method first')
            return None
        else:
            ols_instance = self.ols_gap_decomposition
            print('Adjusted r-square....... ', ols_instance.rsquared_adj) 
        #Inverts group codification
        df_to_explain=self.df_significant[self.df_significant[self.bifurcate_encoded]==1].drop([self.salary], axis=1 )
        df_to_explain[self.bifurcate_encoded]=df_to_explain[self.bifurcate_encoded].replace(1,0)
        #Salary prediction with group codification inverted
        to_explain_predicted= ols_instance.predict(df_to_explain) 
        reference_avg= self.df_significant[self.salary][self.df_significant[self.bifurcate_encoded]==0].mean() 
        to_explain_avg= self.df_significant[self.salary][self.df_significant[self.bifurcate_encoded]==1].mean()
       # predicted minus real salary
        UnexplainedGAP= to_explain_predicted.mean()-to_explain_avg
        RawGAP= reference_avg - to_explain_avg 
        ExplainedGAP= RawGAP-UnexplainedGAP
        to_explain_predicted_head= self.to_explain_head+'_Predicted'
        oaxaca_results= pd.DataFrame({self.reference_head: [reference_avg], 
                                  self.to_explain_head: [to_explain_avg],
                                  'RawGAP':[RawGAP], 
                                  to_explain_predicted_head: to_explain_predicted.mean(),
                                  'Explained_GAP':[ExplainedGAP], 
                                  'Unexplained_GAP':[UnexplainedGAP], 
                                  'R-square_adj':[ols_instance.rsquared_adj]}).T
        oaxaca_results= oaxaca_results.rename(columns={0:'OaxacaB_Two-Fold'})
        self.oaxaca_results= oaxaca_results
        adjusted_pay_gap= self.pay_gap.copy()
        adjusted_pay_gap['AdjustedGAP']= round(UnexplainedGAP,2)
        adjusted_pay_gap["% AdjustedGAP"]= round(UnexplainedGAP/reference_avg*100,4)
        display(adjusted_pay_gap)
        return self.oaxaca_results
    

    def correlation_matrix (self, df=None):
        '''
        Creates a correlation matrix to check multi collinearity
        '''
        if isinstance(df, pd.DataFrame):
            df_corr= df.copy()
        elif  isinstance(self.df_significant, pd.DataFrame):
            df_corr= self.df_significant.copy()
            print('Correlation with df_significant')
        elif isinstance(self.df_dummy, pd.DataFrame):
            df_corr= self.df_dummy.copy()
            print('Correlation with df_dummy')   
        else:
            df_corr= self.df.copy()
        num_cols= df_corr.columns.tolist()
        corr = df_corr[num_cols].corr()
        # plot the heatmap
        plt.figure(figsize=(15,10))
        sns.heatmap(corr, annot=True, cmap='coolwarm',vmax=1,vmin=-1,
            fmt=".2f",
            xticklabels=corr.columns,
            yticklabels=corr.columns)

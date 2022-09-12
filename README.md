# GenderPayGap
Provides methods to automate the Gender Pay Gap analysis. It performs the exploratory data analysis and data modeling to obtain the Salary Gap Decomposition, the Adjusted Gender Pay Gap and the significant variable coefficients. 

More info at: https://medium.com/@fxangulo/an-employers-module-for-gender-pay-gap-analysis-4a13f61a7df1

# Description
This module provides methods to automate the Gender Pay Gap analysis. 
It performs the exploratory data analysis and data modeling to obtain the Salary Gap Decomposition, 
the Adjusted Gender Pay Gap and the significant variable coefficients. 
# Module content
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

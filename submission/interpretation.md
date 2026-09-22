# Machine Learning Interpretation

The full feature set (volatility metrics + socioeconomic controls) outperformed controls alone in the Random Forest nested cross-validation (66.6% vs. 58.4% mean accuracy, winning in 9 out of 10 repeated CV iterations). However, a strict permutation test found this improvement was not statistically significant at the 5% level (p=0.069), likely limited by the extremely small sample size (n=32 districts). 

Furthermore, this weak ML signal aligns with our linear OLS regression results (Model B), which found that the primary price volatility coefficient for stunting was not statistically significant (p=0.822) after adjusting for controls. Therefore, while the machine learning framework detected some predictive signal in price instability, the result should be read strictly as suggestive and exploratory, not statistically confirmed.

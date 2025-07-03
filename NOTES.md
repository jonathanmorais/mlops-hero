### notes

### EDA
first i made a simple EDA to understand better the data, i could see the shape and size of columns and how many missing values have.I calculated median, assimetry and standard deviation and i noticed a big assimetry (-0.49) to the left for monthly income data, i needed to recalculate to desconsider negative values to make more sense and then i got a better assimetry.

### bivariate analyses

Here the goal was to calculate throyt a statical method the correlation btw two or more variables, in this case the target variable against predicte variables. The result was:

2. NumberOfTime30-59DaysPastDueNotWorse 0.1232 🟡 MODERATE 🔴 ↗️ INCREASED risk
3. NumberOfTimes90DaysLate 0.1111 🟡 MODERATE 🔴 ↗️ INCREASED risk
4. NumberOfTime60-89DaysPastDueNotWorse 0.0938 🟢 WEAK 🔴 ↗️ INCREASED risk
5. NumberOfDependents 0.0467 ⚪ VERY WEAK 🔴 ↗️ INCREASED risk
6. Unnamed: 0 0.0048 ⚪ VERY WEAK 🔴 ↗️ INCREASED risk
7. RevolvingUtilizationOfUnsecuredLines -0.0024 ⚪ VERY WEAK 🔵 ↘️ DECREASES risk
8. NumberRealEstateLoansOrLines -0.0030 ⚪ VERY WEAK 🔵 ↘️ DECREASES risk
9. DebtRatio -0.0033 ⚪ VERY WEAK 🔵 ↘️ DECREASES risk
10. MonthlyIncome -0.0197 ⚪ VERY WEAK 🔵 ↘️ DECREASES risk
11. NumberOfOpenCreditLinesAndLoans -0.0274 ⚪ VERY WEAK 🔵 ↘️ DECREASES risk
12. age -0.1027 🟡 MODERATE 🔵 ↘️ REDUCES risk

obs:
MonthlyIncome still with a very assymetry and with this we could see a weak correlation with target variable;

### feature engineering
The basic about FE it's algorithms whether machine learning or another one, don't understand words instead they understand numbers and then we need to convert in "understandble" way for it.

obs: we make sure in achieve the optimal number of features, because many features can cause things overfiting, noise and curse of dimensionality.
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats

def generate_data(seed=42):
    # Set seed for random state
    np.random.seed(seed)

    # Generate testing data
    x1 = np.linspace(0, 50, 100)
    y1 = 3 + 2 * x1 + np.random.normal(size=x1.shape)

    x2 = np.linspace(50, 100, 100)
    y2 = 10 + 4 * x2 + np.random.normal(size=x2.shape)
    
    return x1, y1, x2, y2

def chow_test(x1, y1, x2, y2):
    # Full Data Set Model
    X = np.concatenate([x1, x2])
    X = sm.add_constant(X)
    Y = np.concatenate([y1, y2])
    model_full = sm.OLS(Y, X).fit()

    # Segmented Data Set Models
    X1 = sm.add_constant(x1)
    X2 = sm.add_constant(x2)
    model_segment1 = sm.OLS(y1, X1).fit()
    model_segment2 = sm.OLS(y2, X2).fit()

    # Chow Test Formula
    N1, K1 = X1.shape
    N2, K2 = X2.shape
    F = ((model_full.ssr - (model_segment1.ssr + model_segment2.ssr)) / (model_segment1.ssr + model_segment2.ssr)) * ((N1 + N2 - 2 * K1) / K1)

    # Calculate p-value
    p_value = 1 - stats.f.cdf(F, K1, (N1 + N2 - 2 * K1))

    return F, p_value

def main():
    x1, y1, x2, y2 = generate_data()
    F, p_value = chow_test(x1, y1, x2, y2)

    print(f"Chow Test: \nF-statistic: {F} \nP-value: {p_value}")
    return


if __name__ == "__main__":
    main()

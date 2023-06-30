import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Generate random correlation data
def generate_data():
    np.random.seed(42)
    data = np.random.randn(5, 5)
    corr_data = np.corrcoef(data.T)
    return corr_data

# Create a correlation matrix plot with Matplotlib
def matplotlib_corr_matrix(data):
    fig, ax = plt.subplots()
    cax = ax.matshow(data, cmap='coolwarm')
    fig.colorbar(cax)
    
    # Add correlation values as cell labels
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='black')
    
    plt.xticks(range(data.shape[1]))
    plt.yticks(range(data.shape[0]))
    plt.title("Correlation Matrix - Matplotlib")
    return plt.show()

# Create a correlation matrix plot with Seaborn
def seaborn_corr_matrix(data):
    sns.heatmap(data, cmap='coolwarm', annot=True, square=True)
    plt.title("Correlation Matrix - Seaborn")
    return plt.show()

# Creating a main function to generate the data and run through the plotting approaches. 
def main():
    data = generate_data()
    matplotlib_corr_matrix(data)
    seaborn_corr_matrix(data)
    return

main()

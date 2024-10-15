import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from IPython.display import display, HTML
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def pca_summary(pca, standardised_data, out=True):
    names = ["PC"+str(i) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    a = list(np.std(pca.transform(standardised_data), axis=0))
    b = list(pca.explained_variance_ratio_)
    c = [np.sum(pca.explained_variance_ratio_[:i]) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    columns = pd.MultiIndex.from_tuples([("sdev", "Standard deviation"), ("varprop", "Proportion of Variance"), ("cumprop", "Cumulative Proportion")])
    summary = pd.DataFrame(zip(a, b, c), index=names, columns=columns)
    if out:
        print("Importance of components:")
        display(summary)
    return summary

def printMeanAndSdByGroup(variables, groupvariable):
    data_groupby = variables.groupby(groupvariable)
    print("## Means:")
    display(data_groupby.mean())
    print("\n## Standard deviations:")
    display(data_groupby.std())
    print("\n## Sample sizes:")
    display(pd.DataFrame(data_groupby.apply(len)))

def calclda(variables, loadings):
    # find the number of samples in the data set and the number of variables
    numsamples, numvariables = variables.shape
    # make a vector to store the discriminant function
    ld = np.zeros(numsamples)
    # calculate the value of the discriminant function for each sample
    for i in range(numsamples):
        valuei = 0
        for j in range(numvariables):
            valueij = variables.iloc[i, j]
            loadingj = loadings[j]
            valuei = valuei + (valueij * loadingj)
        ld[i] = valuei
    # standardise the discriminant function so that its mean value is 0:
    ld = scale(ld, with_std=False)
    return ld

#Function to calculate the in group variance
def calcWithinGroupsVariance(variable, groupvariable):
    # Find the unique levels (groups) in the group variable
    levels = sorted(set(groupvariable))
    numlevels = len(levels)  # Number of unique groups
    
    # Initialize counters for the numerator and denominator of the variance calculation
    numtotal = 0
    denomtotal = 0
    
    # Loop through each group (level) to calculate variance within the group
    for leveli in levels:
        # Select data points corresponding to the current group
        levelidata = variable[groupvariable == leveli]
        levelilength = len(levelidata)  # Number of data points in the group
        
        # Calculate the standard deviation for the current group
        sdi = np.std(levelidata)
        
        # Numerator part: Sum of squared deviations for the group
        numi = (levelilength) * sdi**2
        
        # Denominator part: Sum of the number of elements in the group
        denomi = levelilength
        
        # Accumulate the numerator and denominator for all groups
        numtotal = numtotal + numi
        denomtotal = denomtotal + denomi
    
    # Calculate the within-groups variance using the formula
    Vw = numtotal / (denomtotal - numlevels)  # Subtract numlevels to account for degrees of freedom
    
    return Vw

def groupStandardise(variables, groupvariable):
    # find the number of samples in the data set and the number of variables
    numsamples, numvariables = variables.shape
    # find the variable names
    variablenames = variables.columns
    # calculate the group-standardised version of each variable
    variables_new = pd.DataFrame()
    for i in range(numvariables):
        variable_name = variablenames[i]
        variablei = variables[variable_name]
        variablei_Vw = calcWithinGroupsVariance(variablei, groupvariable)
        variablei_mean = np.mean(variablei)
        variablei_new = (variablei - variablei_mean)/(np.sqrt(variablei_Vw))
        variables_new[variable_name] = variablei_new
    return variables_new

def rpredict(lda, X, y, out=False):
    ret = {"class": lda.predict(X),
           "posterior": pd.DataFrame(lda.predict_proba(X), columns=lda.classes_)}
    ret["x"] = pd.DataFrame(lda.fit_transform(X, y))
    ret["x"].columns = ["LD"+str(i+1) for i in range(ret["x"].shape[1])]
    if out:
        print("class")
        print(ret["class"])
        print()
        print("posterior")
        print(ret["posterior"])
        print()
        print("x")
        print(ret["x"])
    return ret

#Function to calculate between group variance
def calcBetweenGroupsVariance(variable, groupvariable):
    # Find the unique levels (groups) in the group variable
    levels = sorted(set(groupvariable))
    numlevels = len(levels)  # Number of unique groups
    
    # Calculate the overall grand mean (mean of all data points)
    grandmean = np.mean(variable)
    
    # Initialize counters for the numerator and denominator of the variance calculation
    numtotal = 0
    denomtotal = 0
    
    # Loop through each group (level) to calculate the variance between groups
    for leveli in levels:
        # Select data points corresponding to the current group
        levelidata = variable[groupvariable == leveli]
        levelilength = len(levelidata)  # Number of data points in the group
        
        # Calculate the mean for the current group
        meani = np.mean(levelidata)
        
        # Numerator part: Weighted sum of squared differences between group mean and grand mean
        numi = levelilength * ((meani - grandmean) ** 2)
        
        # Denominator part: Sum of the number of elements in the group (not directly used here)
        denomi = levelilength
        
        # Accumulate the numerator and denominator for all groups
        numtotal = numtotal + numi
        denomtotal = denomtotal + denomi
    
    # Calculate the between-groups variance using the formula
    Vb = numtotal / (numlevels - 1)  # Adjust denominator by degrees of freedom (numlevels - 1)
    
    return Vb

#Calculate separation between groups

def calcSeparations(variables, groupvariable):
    # Loop through each variable in the dataset
    for variablename in variables:
        # Extract the data for the current variable
        variablei = variables[variablename]
        
        # Calculate the within-groups variance for the current variable
        Vw = calcWithinGroupsVariance(variablei, groupvariable)
        
        # Calculate the between-groups variance for the current variable
        Vb = calcBetweenGroupsVariance(variablei, groupvariable)
        
        # Calculate the separation as the ratio of between-groups variance to within-groups variance
        sep = Vb / Vw
        
        # Print the results for the current variable
        print("variable", variablename, "Vw=", Vw, "Vb=", Vb, "separation=", sep)
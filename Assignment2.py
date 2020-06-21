# Zixuan Zeng
# 12/06/2019
# Assignment2 CSC 273

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.cluster as KMeans
from sklearn.cluster import KMeans


def read_file():
    columns = ['ID', 'Primary Type', 'Latitude', 'Longitude', 'Year']
    df = pd.read_csv('Crimes_-_2001_to_present.csv', usecols=columns)
    return df


# Filter the data frame into smaller one based on year range and crime type
def group_df(df, b, e, crime):
    df_year = df.loc[(df['Year'] >= b) & (df['Year'] <= e)]
    print(df_year)  # debug
    df_crime = df_year.loc[(df_year['Primary Type'] == crime)]

    return df_crime

def filter_df(df, b, e):
    df_year = df.loc[(df['Year'] >= b) & (df['Year'] <= e)]
    print(df_year)  # debug
    df_crime = df_year.loc[(df_year['Primary Type'] == "CRIM SEXUAL ASSAULT")]
    df_crime = df_crime.append(df_year.loc[(df_year['Primary Type'] == "FIRST DEGREE MURDER")])
    df_crime = df_crime.append(df_year.loc[(df_year['Primary Type'] == "THEFT")])

    return df_crime

# This function is to compute the means, median, std based on Longitude and Latitude in data frame
def statistics(df):
    df_stats = df.groupby(['Year'])[['Longitude', 'Latitude']].agg(['median', 'mean', 'std', 'var'])
    print(df_stats)  # For debug only
    return df_stats


# This function plots the map with blue dots indicating the crime's location
def display_map(Longitude, Latitude):
    plt.title('Chicago Crime Map (Sexual Assault)')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.scatter(x=Longitude, y=Latitude, s=0.5)
    plt.show()


def display_histogram(df, stats):
    plt.figure()

    # create histogram based on the data
    df.hist(stacked=True, bins='auto', alpha=0.5)

    # get the labels for each histogram chart d = median, m = mean, s = std, v = variance, a = legend
    a = 'Year Group = ' + str(stats.index.values[0])
    d = round(stats.iloc[0][0], 2)
    m = round(stats.iloc[0][1], 2)
    s = round(stats.iloc[0][2], 2)
    v = round(stats.iloc[0][3], 2)

    # str concatenate them together
    mdsv = "median = " + str(d) + ": mean = " + str(m) + ": std = " + str(s) + ": var = " + str(v)

    # add title and labels for the histogram
    plt.xlabel(mdsv)
    plt.ylabel('CRIM SEXUAL ASSAULT')
    plt.legend([a], loc='best')
    plt.title('Chicago Crime Data Analysis (Sexual Assault)')

    plt.show()


# The k-means cluster algorithm to train the data: to get the best K value
def KMeans_cluster(X):
    sum_of_squared_euclidean_distance = []  # set up a list to store each center as K increases
    for i in range(1, 11):  # Give enough K value to let elbow method which value is the best
        k_means = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        k_means.fit(X)
        sum_of_squared_euclidean_distance.append(k_means.inertia_)
    plt.title('Elbow Method for K value')
    plt.plot(range(1, 11), sum_of_squared_euclidean_distance)
    plt.xlabel('Number of clusters')
    plt.ylabel('Within cluster centroid')
    plt.show()


# This function is to produce cluster centroid map that shows the K-means result
def plot_cluster_centroid(X):
    k_means = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
    predict_y = k_means.fit_predict(X)
    plt.title('Cluster Centroid')
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], s=100, c='red')
    plt.show()

# Plot the line graph that shows the year over year trend
def plot_years_trend(df):
    graph_array = df.groupby(['Year']).count().to_numpy()
    graph_array = np.delete(graph_array, 0, 1)
    print(graph_array)  # Debug
    plt.plot(graph_array)
    plt.show()

def main():
    df = read_file()
    print(df)  ## for debug only
    df_crime = group_df(df, 2014, 2017, "CRIM SEXUAL ASSAULT")
    print(df_crime)  ## debug
    display_map(df_crime['Longitude'], df_crime['Latitude'])
    position_array = df_crime.loc[:,
                     ['Latitude', 'Longitude']].to_numpy()  # Convert last two columns of data frame into numpy array
    position_array = position_array[~np.isnan(position_array).any(axis=1)]  # To ignore NaN values
    print(position_array)  # debug
    KMeans_cluster(position_array)
    plot_cluster_centroid(position_array)
    stats = statistics(df_crime)
    display_histogram(df_crime, stats)

    # Below function calls are for Graduate student part
    begin_year = input("Enter the begin year you want to analyze those three major crimes in Chicago: ")
    end_year = input("Enter the end year: ")
    user_df = filter_df(df, int(begin_year), int(end_year))
    print(user_df)  # Debug
    plot_years_trend(user_df)

if __name__ == "__main__": main()

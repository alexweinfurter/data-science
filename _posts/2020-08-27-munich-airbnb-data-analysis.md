---
title: "Munich AirBnb Data Analysis"
date: 2020-08-27
excerpt: "Exploratory data analysis of Munich AirBnb data and prediction of AirBnb prices."
---

#WORK IN PROGRESS

## Motivation
AirBnb is one of the most successful platforms for apartment rentals world wide. Due to the high housing prices in San Francisco, the original idea of the AirBnb founders was to allow users offering a free bed to travellers to make some extra money. Nowadays AirBnB not only offers short-term rentals at some hosts private apartment but also hotel rooms, long-term apartment rentals and bookable experiences.

In the following blog post I will analyze AirBnB data from Munich, one of the most famous cities of Europe (especially because of the popular Oktoberfest).

## Objectives
This project aims to get a better understanding of the AirBnB market in Munich.
Therefore I'll try to find answers to the following four questions:

1. Which time is the most expensive / cheapest to travel to Munich?

2. Is booking earlier cheaper than booking short time in advance and how do prices change over time?

3. Is it possible to build a model to automatically suggest a price to hosts?

4. Which factors influence the prices most and how do they influence them?

## Dataset
AirBnb itself doesn't provide open source data, but a platform called Inside Airbnb scrapes the listings on a monthly basis (see http://insideairbnb.com/get-the-data.html). I wrote a small web scraper which downloads all the data files that contain data about AirBnbs in Munich.

Essentially Inside Airbnb provides three different files: 
* Calendar: contains prices plus minimum nights to stay per listing and date
* Reviews: contains reviewer names and comments for each listing
* Listings: contains information about the Airbnb, the host and the price.

## Methodology
To analyze the dataset I follow the CRISP-DM cycle. I start with a quick data exploration, clean the dataset and calculate first moments of the data to answer the first two questions. Afterwards I perform some feature engineering and try to build a pricing model. Later the pricing model will be analyzed to get an idea which features influence the price most.

## Exploratory Data Analysis
To answer the first two questions I first used the calendar dataset to get in impression of the price variation over time. As I assumed the prices are by far the highest during September when the Oktoberfest takes place. Besides there's a weekly seasonality that shows higher prices during the weekend. The AirBnBs tend to be cheapest during spring. The peak during April is probably due to a hugh spring festival in Munich ![Prices](/images/munich-airbnb-data/average_price_per_night.png){:class="img-responsive"}

Since its sometimes cheaper or more expensive to book at certain times of a year I investigated if the average booking price varies over time. Therefore I exluded the booking prices of September to remove the high prices during the Oktoberfest and then calculated the daily average booking prices of all AirBnBs in Munich. ![Price at booking time](/images/munich-airbnb-data/prices_at_booking_time.png){:class="img-responsive"}
In the boxplot you can see that the average rental prices tend to be higher if the booking is made during August or September. So the Oktoberfest-effect still influences the prices for future rentals, although the rental date is not in September when the Oktoberfest takes place. Thus, if you want to travel to Munich, it is probably cheaper to make the booking in another month.


## Data cleaning
The following steps have been neccessary to get a clean listings dataset for further processing:
* Reformat prices
* Drop listings with extremely high or low prices
* Drop features with more than 70% missing values
* Drop listings that are duplicates
* Drop listings with missing geolocation
* Drop inactive listings (no activity in 365 days)

## Feature Engineering

### Transformation of categorical variables
The dataset contains several categorical variables like amenities or the neighbourhood that are transformed into dichotomous variables by using one-hot-encoding.

Especially one-hot-encoding the amenities resulted in a large amount of features. Some amenities have been merged, e.g. "TV", "Cable TV" and "Smart TV" are considered as one feature "TV". Furthermore the total amount of amenities has been calculated.

ANZAHL DER CATEGORICALs EINFÜGEN

### Text-based features
Although reviews and descriptions of AirBnBs probably contain valuable information I haven't performed advanced natural language processing (NLP). I only created two text-based features:
* A binary variable containing information about an existing roof top terracce (since roof top terracces in cities are super cool and could increase the price).
* One possible driver of the price could be the size of the AirBnb. Since the feature "square_feet" is missing in most cases, i tried to get that information from the description or summary feature. I used regulare expressions to search for square meters (and all possible abbreviations of it) to find numbers related to the size.

### Binning of numerical features
There are several features containing rates or review scores. I used binning to build groups of these features since e.g. scores that are below 8/10 points can be considered as bad on AirBnB.

### Location based features
The dataset provides the latitude and longitude of each AirBnb. I've looked up the geolocation of several points of interest in Munich (e.g. where the Oktoberfest takes place, the Hofbraeuhaus and the Marienplatz, which is basically the city center) and calculated the haversine distance (greate circle distance) to these points. Afterwards I've calculated the average distance to all points of interest as additional feature.

### Time based features
The features "host_since", "first_review" and "last_review" represent timestamps which have been used to calculate the elapsed time till today. 

### Feature selection
To reduce the number of features several feature selection steps have been performed:
* Removed all features related to the host neighbourhood - this shouldn't impact the price.
* Manually selected amenities that could influence the price.
* Used correlations to find collinear features and remove highly correlated features.
* Dropped binary features with either true or false occuring very seldom.
* Dropped features with more than 70% of missing values.


## Modeling 
To predict the AirBnB prices I trained a regression model with the XGBoost algorithm. 

Although XGBoost is able to handle missing values, all numerical features but the "size" feature have been imputed with the sklearn Iterative Imputer (which is basically an iterative Bayesian Ridge regression). The "size" feature has been imputed by using a Random Forest Regressor with manually chosen features that could influence the size of the AirBnB. The imputers have only been trained on the training set to avoid data leakage.

The model has been evaluated using 10-fold cross-validation and has an average r-squared of 0.71 on the training set and 0.44 on the test set. Thus the model seems to overfit and only explains 0.44 of the prices variance. 


No feature or target scaling has been used since XGBoost (and tree-based models in general) should be invariant to monotonic transformations like log-transform, etc..

## Model explanation

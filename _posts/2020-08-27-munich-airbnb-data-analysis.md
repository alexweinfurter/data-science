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

## Data cleaning
The following steps have been neccessary to get a clean dataset for further processing:
* Reformat prices
* Drop listings with extremely high or low prices
* Drop features with more than 70% missing values
* Drop listings that are duplicates
* Drop listings with missing geolocation

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
To reduce the number of features several feature selection have been performed:
*  Removed all features related to the host neighbourhood - this shouldn't impact the price.
* Used correlations to find collinear features and remove highly correlated features.
* Dropped features with very low variance.
* Dropped features with more than 70% of missing values.


## Modeling 
No feature or target scaling has been used since XGBoost (and tree-based models in general) should be invariant to monotonic transformations like log-transform, etc..

## Model explanation

# Music_Time_Period_Classification

## Brief Introduction

We believe a fundamental part of everyone’s life is *music*, also we believe that *nostalgia* is a very
strong feeling, so the goal of the study is to create a machine learning model that can correctly
classify songs by decade in order to improve the *listener experience*. Specifically, we want to
help the music industry to create *playlists* or *recommendations systems*, to provide a way to track
the cultural and social influences that changed the music over time and to find recurring features
to predict the next big music movement.
We’ll use a dataset built with the *Spotify API* via the *Spotify* Python library: *Spotipy*.
After creating and cleaning the dataset, we will use multiple classification models to train and
evaluate predictions’ performance. We are going to demonstrate the potential of machine
learning in analyzing music data and offer insights into how popular music changed over time.

## Creation of the dataset

To collect the data we implemented a Spotipy script module in Python. Spotipy
grants access to all the Spotify songs. We wrote a scalable script that created a
relatively big dataset (9425 songs) for the six decades 60s, 70s, 80s, 90s, 00s, 10s.
In order to gain access to Spotify via the Spotipy API, the first thing we need to do is to create an
account Spotify for Developers to get the credentials for the Authentication. After retrieving a
*CLIENT_ID* and an *SECRET_CLIENT_ID*, we can connect to Spotify via the Spotipy library in
Python, by creating the Spotify python object.

In the *report* and in the *ppt* is possible to find the exploratory and the explation of all the variables in the dataset, the variable we will try to predict is the *decade* label.

Since near decades share more or less similar features, we passed from 6 decades to three groups of twenty years:
X-Generation (60s plus 70s), Y-Generation (80s plus 90s) and Z-Generation (00s plus 10s).

Some plots for the Exploratory Data Analysis

To plot the distribution of each numerical variable within each 'decade' group, we can create a set
of histograms or density plots for each feature. This will allow us to visualize how the values of
each feature are distributed across different 'decade' categories.

<img width="785" alt="plot1" src="https://user-images.githubusercontent.com/113529675/271364418-e5bf8864-6619-4ee5-9538-8a4eeb5b7024.png"> 

In the box plots we are comparing the X-generation (60s-70s) to the Z-generation (80s-90s) and
the Z-generation (00s-10s) for the features ‘danceability’, ‘energy’, ‘loudness’, ‘acousticness’,
‘speechiness’, and ‘valence’, we can observe distinct differences.
● Danceability:
  ○ Z Generation (2000-2020): Songs are more danceable, likely due to the
      prevalence of dance-oriented genres.
  ○ X and Y Generations (60s-70s-80s-90s): Songs may be less danceable, offering
      a broader range of genres.
● Energy:
  ○ Z Generation (2000-2020): Similar energy levels as the Y generation, known for
      dynamic and high-energy music.
  ○ X Generation (60s-70s): Possibly lower energy levels compared to Z and Y
      generations, featuring diverse genres.

Other comments are in the report file.

To plot the distribution of each numerical variable within each 'decade' group, we can create a set
of histograms or density plots for each feature(now just 3 are displayed). This will allow us to visualize how the values of
each feature are distributed across different 'decade' categories.

<img width="785" alt="plot1" src="https://user-images.githubusercontent.com/113529675/271364576-722d7f5a-b039-47eb-9b9c-74299a302457.png"> 

## Models

In our analysis, we tackled the class imbalance issue by using *Random Over-Sampling* (ROS)
with *SMOTE* to balance the dataset(was not so unbalanced, but the Z class has less elements).
We then tried different classification methods, evaluating their performance using both accuracy
and balanced accuracy metrics. This approach helped us choose the best model, which turned
out to be the *Random Forest classifier*. Despite some other methods having similar accuracy
values, Random Forest demonstrated superior overall performance and robustness.

We examined the feature importances obtained from the Random Forest classifier, which ranks
the features based on their significance in predicting the decade labels. The bar plot reveals the
following ranking of features from the most to the least.

<img width="785" alt="plot1" src="https://user-images.githubusercontent.com/113529675/271364688-101f8548-0013-497c-be84-ac9603827929.png"> 

By understanding the order of feature importances, we can gain valuable insights into the distinct
musical characteristics that define each decade and use this knowledge to better interpret our
model's predictions.

# Used technologies

![Python](https://a11ybadges.com/badge?logo=python)
![Spotify](https://a11ybadges.com/badge?logo=spotify)

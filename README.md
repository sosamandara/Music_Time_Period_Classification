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

<img width="785" alt="plot1" src="https://private-user-images.githubusercontent.com/113529675/269432097-f873f6be-363a-4733-b4c7-9805a73ee45a.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE2OTUyNDgzMTEsIm5iZiI6MTY5NTI0ODAxMSwicGF0aCI6Ii8xMTM1Mjk2NzUvMjY5NDMyMDk3LWY4NzNmNmJlLTM2M2EtNDczMy1iNGM3LTk4MDVhNzNlZTQ1YS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMwOTIwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMDkyMFQyMjEzMzFaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT00NTE0ZTkwYzdiNmM3MWQzNDU5NmFiYTFhZWE0NjhhNTZiZDgzNmZkZjM2Y2QxNDE3ZjU1ZWRiZDFlNjgzZjBkJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.17xCQBDPgK0aO-qGTbo8ByB8odlt6e3FMlVKQWPtuEI"> 

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

<img width="785" alt="plot1" src="https://private-user-images.githubusercontent.com/113529675/269431381-e653b308-fc84-4d3e-885f-4db968221ef7.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE2OTUyNDgwMzUsIm5iZiI6MTY5NTI0NzczNSwicGF0aCI6Ii8xMTM1Mjk2NzUvMjY5NDMxMzgxLWU2NTNiMzA4LWZjODQtNGQzZS04ODVmLTRkYjk2ODIyMWVmNy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMwOTIwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMDkyMFQyMjA4NTVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT03OWZjMjk4ZTliZTUwNDQzMmVmODhkOGMyM2ZjZmZhMjQ3NGMyZTE5MjlmYTQ5MDMxNmYyMDFjODM0NTU1ZTI2JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.gXCkYLI-gMZ3xpi1_z3zS2KQytAT0YgIzKUnhZeK2hU"> 

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

<img width="785" alt="plot1" src="https://private-user-images.githubusercontent.com/113529675/269432202-27f06508-59b7-4301-be64-e40b0e3b4f38.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE2OTUyNDgzNjUsIm5iZiI6MTY5NTI0ODA2NSwicGF0aCI6Ii8xMTM1Mjk2NzUvMjY5NDMyMjAyLTI3ZjA2NTA4LTU5YjctNDMwMS1iZTY0LWU0MGIwZTNiNGYzOC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMwOTIwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMDkyMFQyMjE0MjVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT00MzQ3YzNmNjhmOWY1MDUwMDBmYjQ5YTgwZmFkMWU1MDlkMzY5MWI3NTA0MGIyOWY4OWQyZWY1M2YyNjJhZWRhJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.HTgFXSr3Bb4EWI4Qy0nxlBoTBIx2-3ex1JwwJlWsXjM"> 

By understanding the order of feature importances, we can gain valuable insights into the distinct
musical characteristics that define each decade and use this knowledge to better interpret our
model's predictions.

# Used technologies

![Python](https://a11ybadges.com/badge?logo=python)
![Spotify](https://a11ybadges.com/badge?logo=spotify)

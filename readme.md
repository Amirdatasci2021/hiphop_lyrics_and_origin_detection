# Machine Learning and Source recognition:
## Internet Hip hop Lyrics and the depth of NLP

### Introduction

Meet Hiphop, the first true artistic expression was born and raised during the third industrial revolution. So while we may not associate hip hop with what has brought semiconductors, mainframe computing, personal computing, and the Internet—the digital revolution. It is also the only art form to transcend into the beginning of the next phase of dramatic technological expansion and social change—the *Fourth Industrial Revolution*.

By the time music became mainstream it had taken on the attributes of show business and Hollywood. It was also fused with what always has been provactive and alluring to a general audience: *sex and violence*. Much of what is produced is done so for the purpose of making money. There are inherit similarities in words, slang and aphormisms. **So are all hip hop lyrics essentially the same regardless of who delivers or authors them?**

### Executive Summary

The commodification of Hip Hop is pretty widespread; music as means to prosperity means the sacrifice of artistic integrity and "manufactured" personas are rampant. The commodification is also true for the transcription of lyrics by private individuals who are unpaid and the taking of that essentially gifted contribution and turning it into a commodity by web sites like Lyrical Genius.

**Problem statement:**
Can Natural Language Processing predict what artist is associated with a set of  transcribed lyrics?

**Secondary question:**
How can we account for where that success is taking place and what are the things that allow the machine to make an accurate prediction? 

**Third question:** 
What makes these lyrics difficult to classify and how is similar to NLP challenges in real world applications?

| Data Source 	| Url 	|
|-	|-	|
| Hip-Hop Encounters Data Science 	| https://www.kaggle.com/rikdifos/rap-lyrics 	|

**Specail thanks to:* Seanny for sharing this dataset


**example:**

> "you try to plant somethin in the conrete yknowhatimean if it grow and
> the and the rose petal got all kind of scratches and marks you not gon
> say damn look at all the scratches and marks on the rose that grew
> from concrete you gon be like damn a rose grew from the concrete same
> thing with me yknahmean i grew out of all of this"

 - Firstly the ‘somethin’ or something is simple there is a ‘g’ missing.
   A  machine could recognize that very easily. We are able to determine
   it’s part of speech and can simply overlook the alteration. However,
   what if it was “nuffin” and ‘nothing’? **Can the machine keep up?**
 - Secondly is something like "yknowhatimean" or "nahmean" which is the
   disambiguation and modification of "you know what I mean" or "gon"
   which is "gonna" a disambiguation of "going to". **How would a machine
   interpret that?**
 - Lastly are the countless ways that profanity is written or
   transcribed in text and not only these lyrics but on the web and
   social moderated communities. **How can a machine recognize those?**

One thing that I have observed in the language that people use as of late is the word "Im (a) finna" as an example. This is a disambiguation of the phrase "I am fixin' to" which is extremely provincial and not something that I have heard used in a place like New York. However, now I can observe this expression making its way through a large amount of comments and responses on social media. **The question is how does a machine process something like that and more importantly how long does it take it to understand the use, the context and its multitude of intentional misspellings? Or are we over analyzing the problem because we lack confidence in the machine learning model?**

**Natural Language Processing (NLP)** is a branch of **Artificial Intelligence (AI)** that studies how machines understand human language. Its goal is to build systems that can make sense of text and perform tasks like translation, grammar checking, or topic classification. The process really can work in both supervised and unsupervised training. The information provided is referred to as the training data and the text provided as the corpus. What we predict on is the testing data. 

Modeling wise we will be using **Scikit-learn**  which is a free software machine learning library for the **Python** programming language. It features various classification, regression and clustering algorithms.With smaller samples, **Bernoulli Naive Bayes** gives more accurate and precise predictions as opposed to other models. I choose it for speed and the fact that it really gives quick predictions. In relation to the text data here we have selected Count Vectorizer as our tokenizer. TfidfVectorizer() assigns a score while CountVectorizer() counts and after careful evaluation of both. We will be using **CountVectorizer()** as we do not need the scoring component. 

**Other Models used: Support Vector Classifiers, KNClassifier**

**Table of Contents**

| File | Link|
|-	|-	|
| Intial Eda	|  	|
| Secondary Eda 	|  	|
| Test Sample Maker	|  	|
| Binary 01: Naive Bayes 	|  	|
| Multiclass 02: Multinomial Naive Bayes 	|  	|
| 	|  	|


**Process:**

**Data Prep**
The data is collectively broken down to a line by line level. There are two groups one is short length that are 36 classes and 4 that are longer lyric text blocks. The control group is scraped directly from an earlier version of this project.  Once properly cleaned and prepared the complete export file is provided to a Test Sample Maker which will pull out four random artist and their lyrics. The file maker can run a seperate py file that can create as many samples as needed.

The largest I have worked with is: 2469 combinations

The files are named for the artists.

**Modeling process**

The files are extracted and converted to a binary target and other. The sequence runs with the particular model using the first scenrio to create a grid search. The best parameters are then used to run the model. The results are then filtered into a Pandas dataframe for further examination. The idea is that we are running multiple combinations of partial targets and achieving an understanding of the individual classifications that are taking place. 

Our problem statement was:
Can Natural Language Processing predict what artist is associated with a set of transcribed lyrics?
If our null hypothesis was: "NLP can not tell the difference between one rapper's internet transcribed lyrics from another rappers internet transcribed lyrics"
and
If our alternative hypothesis was: "NLP can tell the difference between one rapper from another internet transcribed lyrics"
In real world data application, it can be puzzling whether a binary decision problem should be formulated as hypothesis testing or a binary classification.
So here is what formuliacly what Accuracy is equal to:
(TN + TP)/(TN+TP+FN+FP) = (Number of correct assessments)/Number of all assessments)
in other words (Correct Others + Correct Targets)/(Correct and Inncorrect for both the others and the target groups)
So yes NLP can tell the difference between one rapper from another internet transcribed lyrics
However, the true positive rate(TPR) against false positive rate (FPR) can not be measured, where TPR= TP/(TP+FN) And FPR = FP/(FP+TN) because we have neglected everything but the true positive. This is a high specificity test. We have pretty much sacrificed sensitivity for the sake of a quick answer.

**Results**
Binary performance was strong with a few cases of performing below .5 but all in all the model predicted well beyond a random guess.
Multinomial Predictions for the true positive we much weaker and so was accuracy while the larger average was above .5 it was still not impressive. The only targets to consistantly have strong predictions were our control group. 


**Recommendations**

1. Explore a better way storing information: MongoDB is a great tool that works as a great alternative to a relational database and also CSV files.Instead of outputting a hundred files you would execute a queries which are relatively fast depending on your internet connection. 
2. Explore different metrics: Instead of focusing on accuracy explore the True Positive Rate and include a deeper analysis on Sensitivity.
3. Explore alternative sources of text that have a stronger uniformity and less of a variety in dealing with the same “words”.
4. Models: Be warned! Logistic Regression, Random Forest and Decision Trees will take a very long time not for one iteration of this experiment both for grid search and turning out results. 
One to try:
XGBoost and Tune the Class Weighting Hyperparameter)
5. Running each individual sequence through RNN is not far fetched for maybe 20-30 epochs. I would recommend if you do this that you use MongoDB as it offers a better alternative to clunky csvs




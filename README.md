# Sentiment Analysis
Two different models for Twitter sentiment analysis - CNN and Naive Bayes Classifier

## DATASET

SOURCE: http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip

- the original dataset was merged in order to manually split it by utilizing the sklearn.TrainTestSplit
- contains over 1.6 million annotated Tweets with positive, negative and neutral sentiments
- for the purposes of our model, the 169 occurrences with neutral sentiment have been removed
- the dataset is shown to be highly balanced, with exactly 1:1 relationship between positive and negative values

## DATA PREPROCESSING

- in order to practice manual implementation, the preprocessing was done by manually removing junk via regex and NLTK's Stopwords, rather than directly doing it with the Tokenizer 
- based on Camacho-Collados & Taher Pilhevar's findings in their paper _On the Role of Text Preprocessing in Neural Network Architectures:
An Evaluation Study on Text Categorization and Sentiment Analysis_ I decided to skip on stemming/lemmatization and proceed directly to tokenization
- two different tokenizers were used, one for each of the models, since the output of tf's Tokenizer is not supported by Multinomial Naive Bayes directly, and thus it was neccessary to use the Bag-of-Words model for it. This, however, excludes realistic comparisons of performance between the models, but as that was not the primary goal, I do not consider it a hinderance. This is expected to be a recurring problem if, in the future, I decide to implement additional models which expect different inputs

## MULTINOMIAL NAIVE BAYES
- the Multinomial variation was chosen based on the reports that it performs better than Gaussian in classification tasks - this is yet to be tested personally and on this specific dataset
- the model performed reasonably well, reaching 74% accuracy, the value consistent through both precision and recall metrics. It also correctly predicted the sentiment of one test Tweet

## CONVOLUTIONAL NEURAL NETWORK
- for the CNN model, a relatively simple architecture is used containing two convolutional layers and one dense layer
- as the model was overfitting massively, batch normalization and an increasing droput rate were introduced to get it under control. For the same reason, a few layers were dropped from the original architecture, and all the hyperparameters were decreased in size, for example, the Embedding Dimension started out as 128, and is now 32. In order to decrease training time, batch size started out as 1000, however, that also played a massive role in overfitting, so it was decreased to 100
- after 5 epochs, the model's accuracy and loss stablilize at around 80% and 0.46 respectively, and training is interrupted with an EarlyStopping mechanism
- after the evaluation on the test set, the model performed marginally better than the Naive Bayes, so the question arises whether the significantly longer training time of the CNN is worth just a couple more % points in accuracy

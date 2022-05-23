#Small example of Naive bayes's classifier using sample tweets 
#to detect suicide or not 
sample <- read.csv("sample.csv")

#Cleaning our dataset
library(tm)
library(e1071)
sample$Stressed = factor(sample$Stressed)
corpus = Corpus(VectorSource((sample$text)))
inspect(corpus)
corpus_clean = tm_map(corpus, content_transformer(tolower))
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)
inspect(corpus_clean)
dtm = DocumentTermMatrix(corpus_clean)
as.matrix(dtm[1:7,])
Train = sample[1:5,]
Test = sample[6:7,]
dtm_train = dtm[1:5,]
dtm_test = dtm[6:7,]
corpus_train = corpus_clean[1:5]
corpus_test = corpus_clean[6:7]
train = DocumentTermMatrix(corpus_train)
as.matrix(train)
test = DocumentTermMatrix(corpus_test)
convert_counts = function(x){
  x = ifelse(x>0, 1, 0)
  x = factor(x,levels = c(0,1), labels = c("FALSE","TRUE"))
  return(x)
}
train = apply(dtm_train, MARGIN = 2, convert_counts)
test = apply(dtm_test, MARGIN = 2, convert_counts)

df_test = as.data.frame(train)
df_test$class = Train$Stressed
df_test = df_test[,-12]

classifyer = naiveBayes(train,Train$Stressed, laplace = 1)

test_pred = predict(classifyer,test, type = "raw")
test_pred



#Big DATA: Suicide Detection
Suicide_Detection <- read.csv("Suicide_Detection.csv")
suicide = Suicide_Detection[,2:3]
set.seed(123)
included_ind = sample(1:nrow(suicide), floor(0.1*nrow(suicide)))
suicide = suicide[included_ind, ]
prop.table(table(suicide$class))
sum(suicide$class == "") 
suicide$missing = ifelse(suicide$class == "", NA, 0 )
suicide = na.omit(suicide)
suicide = suicide[,1:2]
prop.table(table(suicide$class))
table(suicide$class)

#Text cleaning code 
#Get rid of special characters
suicide$text = gsub("[^0-9A-Za-z///' ]","" , df2$bookDesc ,ignore.case = TRUE)

library(tm)
corpus = Corpus(VectorSource((suicide$text)))#Create a corpus for the tweets in suicide data set
inspect(corpus[1:5])#Looking at our documents
corpus = tm_map(corpus, tolower) #Clean data by making all letters in each document to lowercase
corpus = tm_map(corpus, removeWords, stopwords('english'))#Clean data by removing the most frequent English words in the data 
corpus = tm_map(corpus, removePunctuation)#Clean data by removing any punctuation in each document
corpus = tm_map(corpus, removeNumbers)#Clean Data by removing any numbers in each document
corpus_clean = tm_map(corpus,stripWhitespace)#Clean data by removing large white spaces 
inspect(corpus_clean[50:100])#Looking at our documents after cleaning the

dtm = DocumentTermMatrix(corpus_clean)
as.matrix(dtm[1:5,1:5])

#Splitting the data
set.seed(123)
train_ind = sample(1:nrow(suicide), floor(0.8*nrow(suicide)))
Train_raw = suicide[train_ind,]
Test_raw= suicide[-train_ind, ]
#Splitting DTM
dtm_train = dtm[train_ind, ]
dtm_test = dtm[-train_ind, ]
#Splitting corpus 
corpus_train = corpus_clean[train_ind]
corpus_test = corpus_clean[-train_ind]


#Obtaing a word cloud for each both levels for training
library(wordcloud)
train_suicidal_ind = which(Train_raw$class == "suicide", arr.ind = TRUE, useName = FALSE)
corpus_train_suicidal = corpus_clean[train_suicidal_ind]
corpus_train_nonsuicidal = corpus_clean[-train_suicidal_ind]

wordcloud(corpus_train_suicidal, min.freq = 50, max.words =300, colors = brewer.pal(8,'Dark2'))
wordcloud(corpus_train_nonsuicidal, min.freq = 50, max.words = 300, colors = brewer.pal(8,"Dark2"))


convert_counts = function(x){
  x = ifelse(x>0, 1, 0)
  x = factor(x,levels = c(0,1), labels = c("False","True"))
  return(x)
}



#use dictionary of training data set only
dtm_train = DocumentTermMatrix(corpus_train, list(dictionary = c(findFreqTerms(dtm_train,25))))
dtm_test = DocumentTermMatrix(corpus_test, list(dictionary = c(findFreqTerms(dtm_train,25))))
nb_train = apply(dtm_train, MARGIN = 2 , convert_counts)
nb_test = apply(dtm_test, MARGIN = 2, convert_counts)

library(e1071)
nb_model = naiveBayes(nb_train,Train_raw$class,laplace = 1)

nb_test[1:5,1:5]


preds = predict(nb_model,nb_test,type = "class")



#Obtaining predictions 
library(gmodels)
CrossTable(preds, Test_raw$class, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, porp.c = FALSE, digits = 2,
           dnn = c('predicted', 'actual'))






####Comparison of Models- Classification tree
library(rpart)

#Changing to a data frame
df_train = as.data.frame(as.matrix(nb_train))
df_test = as.data.frame(as.matrix(nb_test))
temp_col_names = colnames(df_train)
colnames(df_train) = paste("V", 1:ncol(df_train), sep = "")
colnames(df_test) = paste("V", 1:ncol(df_test), sep = "")
df_train$class = Train_raw$class
df_test$class = Test_raw$class
reg_tree = rpart(class ~ ., data = df_train, method = "class", xval = 10, cp = 0.001)
prob = predict(reg_tree, newdata = df_test, type = "class")
table(prob, Test_raw$class)




temp_col_names = colnames(nb_train)
colnames(nb_train) = paste("V", 1:ncol(nb_train), sep = "")

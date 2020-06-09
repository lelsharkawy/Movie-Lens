#Load the following libraries to allow access for all functions used in the following code
library(dplyr)
library(stringr)
library(tibble)
library(readr)
library(tidyr)
library(caret)
library(data.table)

#Set the number of digits shown during print
options(pillar.sigfig=5)
options(digits=5)

#Download the data required for this project and store it temporarily in dl
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

#Unzip the data downloaded and make it tidy by restructuring and renaming the columns
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")

# Split data into 2 sets, 90% into EDX and 10% into temp (which will later be Validation)
set.seed(1, sample.kind ="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

#Split the EDX dataset into train and testing (which will later be test) set to allow for cross validation in future sets, with the testing set containing 20% of data
set.seed(1, sample.kind ="Rounding")
test_ind <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_ind,]
testing <- edx[test_ind,]

# Make sure userId and movieId in test set are also in train set
test_set <- testing %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set back into training set
remov <- anti_join(testing, test_set)
train_set <- rbind(train_set, remov)

#Remove all files which will no longer be used, to clear space
rm(dl, ratings, movies, test_index, temp, movielens, removed, test_ind,testing, remov)

#Creat RMSE function which will be used to analyze all methods
RMSE<-function(tr_rate,pr_rate){
  sqrt(mean((tr_rate-pr_rate)^2))
}

#Create a data frame which will store all RMSE information for the various methods
rmse_df<- data.frame(method=c("Average Rating", "Movie Specific Effect", "User Specific Effect", "Movie and User Effect", "Regularization for Movie Effect", 
                                 "Regularization for User Effect", "Regularization for Movie and User Effect"), RMSE_edx=c(1:7),  RMSE_validation=c(1:7))

#Method 1, predicting based on average rating for all movies
mu<-mean(edx$rating)
rmse_df[1,2]<-RMSE(edx$rating, mu)
rmse_df[1,3]<-RMSE(validation$rating,mu)

#Method 2, predicting based on the average and movie specific effect
movie_avg<-edx%>%group_by(movieId)%>%summarize(b_i=mean(rating-mu))
pr_e<-edx%>%left_join(movie_avg, by="movieId")%>%mutate(pred=mu+b_i)%>%.$pred
rmse_df[2,2]<-RMSE(edx$rating,pr_e)
pr_r<-validation%>%left_join(movie_avg, by="movieId")%>%mutate(pred=mu+b_i)%>%.$pred
rmse_df [2,3]<-RMSE(validation$rating, pr_r)

#To save space, we will clean the environment after each method
rm(movie_avg,pr_e,pr_r)

#Method 3, predicting based on the average and user specific effect
user_avg<-edx%>%group_by(userId)%>%summarize(b_u=mean(rating-mu))
pr_e<-edx%>%left_join(user_avg, by="userId")%>%mutate(pred=mu+b_u)%>%.$pred
rmse_df[3,2]<-RMSE(edx$rating, pr_e)
pr_r<-validation%>%left_join(user_avg, by="userId")%>%mutate(pred=mu+b_u)%>%.$pred
rmse_df[3,3]<-RMSE(validation$rating, pr_r)

#To save space
rm(user_avg,pr_e,pr_r)

#Method 4, predicting based on the average and both movie and user specific effect
movie_avg<-edx%>%group_by(movieId)%>%summarize(b_i=mean(rating-mu))
user_avg<-edx%>%left_join(movie_avg,by="movieId")%>%group_by(userId)%>%summarize(b_u=mean(rating-mu-b_i))
pr_e<-edx%>%left_join(movie_avg, by="movieId")%>%left_join(user_avg, by="userId")%>%mutate(pred=mu+b_i+b_u)%>%.$pred
rmse_df[4,2]<-RMSE(edx$rating, pr_e)
pr_r<-validation%>%left_join(movie_avg, by="movieId")%>%left_join(user_avg, by="userId")%>%mutate(pred=mu+b_i+b_u)%>%.$pred
rmse_df[4,3]<-RMSE(validation$rating, pr_r)

#To save space
rm(movie_avg, user_avg,pr_e,pr_r)

#Method 5, predicting based on the Regularization for movie specific effect
#Regularization done through lambda (li) cross validation using test set
li<-seq(0,10,0.25)
RMSE_lis<-sapply(li,function(x){
  sum_i<-train_set%>%group_by(movieId)%>%summarize(s_i=sum(rating-mu),n_i=n(), bi=s_i/(n_i+x))
  pr_test<-test_set%>%left_join(sum_i, by="movieId")%>%mutate(pred=mu+bi)%>%.$pred
  return(RMSE(test_set$rating, pr_test))
})

#Visualize the RMSE changes based on lambda (li)
#plot(li,RMSE_lis) 

#Choose lambda (li) pretaining to the minimum RMSE 
#min(RMSE_lis)
L_i<-li[which.min (RMSE_lis)]

#Use lambda found from cross validation (L_i) to calculate and predict for regularized movie specific effect
sum_i<-edx%>%group_by(movieId)%>%summarize(s_i=sum(rating-mu),n_i=n(), bi=s_i/(n_i+L_i))
pr_test<-edx%>%left_join(sum_i, by="movieId")%>%mutate(pred=mu+bi)%>%.$pred
rmse_df[5,2]<-RMSE(edx$rating, pr_test)
pr_r<-validation%>%left_join(sum_i, by="movieId")%>%mutate(pred=mu+bi)%>%.$pred
rmse_df[5,3]<-RMSE(validation$rating, pr_r)

#To save space
rm (sum_i, pr_test, pr_r)

#Method 6, predicting based on the Regularization for user specific effect
#Regularization done through lambda (lu) cross validation using test set
lu<-seq(0,10,0.25)
RMSE_lus<-sapply(lu,function(x){
  sum_u<-train_set%>%group_by(userId)%>%summarize(s_u=sum(rating-mu),n_u=n(), bu=s_u/(n_u+x))
  pr_test<-test_set%>%left_join(sum_u, by="userId")%>%mutate(pred=mu+bu)%>%.$pred
  return(RMSE(test_set$rating, pr_test))
})

#Visualize the RMSE changes based on lambda (lu)
#plot(lu,RMSE_lus) 

#Choose lambda (lu) pretaining to the minimum RMSE 
#min(RMSE_lus)
L_u<-lu[which.min (RMSE_lus)]

#Use lambda found from cross validation (L_u) to calculate and predict for regularized user specifc effect
sum_u<-edx%>%group_by(userId)%>%summarize(s_u=sum(rating-mu),n_u=n(), bu=s_u/(n_u+L_u))
pr_test<-edx%>%left_join(sum_u, by="userId")%>%mutate(pred=mu+bu)%>%.$pred
rmse_df[6,2]<-RMSE(edx$rating, pr_test)
pr_r<-validation%>%left_join(sum_u, by="userId")%>%mutate(pred=mu+bu)%>%.$pred
rmse_df[6,3]<-RMSE(validation$rating, pr_r)

#To save space
rm(sum_u, pr_test, pr_r)

#Method 7, predicting based on the Regularization for both movie and user specific effect
#Regularization done through lambda (l) cross validation using test set
l<-seq(0,10,0.25)
RMSE_ls<-sapply(l,function(x){
  sum_i<-train_set%>%group_by(movieId)%>%summarize(s_i=sum(rating-mu),n_i=n(), bi=s_i/(n_i+x))
  sum_u<-train_set%>%left_join(sum_i, by="movieId")%>%group_by(userId)%>%summarize(s_u=sum(rating-mu-bi),n_u=n(), bu=(s_u)/(n_u+x))
  pr_test<-test_set%>%left_join(sum_u, by="userId")%>%left_join(sum_i, by="movieId")%>%mutate(pred=mu+bu+bi)%>%.$pred
  return(RMSE(test_set$rating, pr_test))
}) 

#Visualize the RMSE changes based on lambda (l)
#plot(l,RMSE_ls) 

#Choose lambda (l) pretaining to the minimum RMSE 
#min(RMSE_ls)
L<-l[which.min (RMSE_ls)]

#Use lambda found from cross validation (L) to calculate and predict for regularized movie and user specifc effects
sum_i<-edx%>%group_by(movieId)%>%summarize(s_i=sum(rating-mu),n_i=n(), bi=s_i/(n_i+L))
sum_u<-edx%>%left_join(sum_i, by="movieId")%>%group_by(userId)%>%summarize(s_u=sum(rating-mu-bi),n_u=n(), bu=(s_u)/(n_u+L))
pr_test<-edx%>%left_join(sum_u, by="userId")%>%left_join(sum_i, by="movieId")%>%mutate(pred=mu+bu+bi)%>%.$pred
rmse_df[7,2]<-(RMSE(edx$rating, pr_test))
pr_r<-validation%>%left_join(sum_u, by="userId")%>%left_join(sum_i, by="movieId")%>%mutate(pred=mu+bu+bi)%>%.$pred
rmse_df[7,3]<-RMSE(validation$rating, pr_r)

#To save space
rm(sum_i, sum_u, pr_test, pr_r)

#Convert rmse data frame to a table
RMSE_results<-as_tibble(rmse_df)


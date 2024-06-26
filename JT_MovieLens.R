# Aauthor: Jozsef Toth
# Title: MovieLens Project (Capstone)
# Date: May 8, 2024

##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 240)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


##########################################################
# Create train/test datasets and helper functions
##########################################################

#Create train and test sets
set.seed(1)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]
nrow(train_set)
nrow(test_set)

#Helper function to calculate RMSE and accuracy
#For accuracy we use the half rounded numbers (0,0.5,1,1.5,2,2.5, etc)
#For RMSE we use the caret function caret::RMSE
Evaluate_Model <- function(y_hat,model_name="Undefined",y=test_set$rating) {
  accuracy <- mean(Round_Rating(y_hat) == y)
  rmse <- RMSE(y_hat,y)
  data.frame(model_name,rmse,accuracy)
}

#Helper function to half round number to closest possible rating
#Possible rating values : (0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5)
Round_Rating <- function(y) {
  y_rounded <- round(y*2)/2
  y_rounded[y_rounded<0] <- 0
  y_rounded[y_rounded>5] <- 5
  y_rounded
}


##########################################################
# Characteristic of the the datasets and data preparation
##########################################################

#edx dataset in general
nrow(edx)
colnames(edx)
str(edx)

#test dataset
nrow(test_set)

#Characteristic of the train dataset
nrow(train_set)

userId_count <- train_set %>% select(userId) %>% distinct() %>% nrow()
userId_count

movieId_count <- train_set %>% select(movieId) %>% distinct() %>% nrow()
movieId_count

genres_count <- train_set %>% select(genres) %>% distinct() %>% nrow()
genres_count

#Ratings in general stats
test_set %>%
  summarise(descr="general rating stats",min=min(rating),max=max(rating),
            mean=mean(rating),median=median(rating)) %>% 
  as.data.frame(.)

#Ratings in general histogram
test_set %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.5, fill = "cyan", col = "black") +
  xlab("rating")+ylab("number of ratings")+
  scale_x_continuous(breaks = c(0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5))+
  ggtitle("Ratings historgram")

#Rating counts for movies
movie_stats <- test_set %>%
  group_by(movieId) %>%
  summarize(rating_count=n()) %>%
  summarise(descr="Rating counts for movies",
            min=min(rating_count),max=max(rating_count),
            mean=mean(rating_count),median=median(rating_count))

movie_stats

#Rating counts for movies
test_set %>%
  group_by(movieId) %>%
  summarize(rating_count=n()) %>%
  arrange(desc(rating_count)) %>%
  mutate(nr=row_number()) %>%
  ggplot(aes(nr,rating_count)) +
  geom_bar(stat="identity", col = "black") +
  geom_hline(aes(yintercept=as.numeric(movie_stats[1,4]),color="red")) +
  guides(color = "none") +
  xlab("movies")+ylab("number of ratings")+
  theme(axis.text.x = element_blank()) +
  ggtitle("Rating counts for movies")

#Rating distribution for movies histogram
test_set %>%
  group_by(movieId) %>%
  summarize(avg_score=mean(rating)) %>%
  ggplot(aes(avg_score)) +
  geom_histogram(binwidth = 0.1, fill = "cyan", col = "black") +
  xlab("average rating")+ylab("number of movies")+
  ggtitle("Rating distribution for movies")

#Rating counts for users stats
user_stats <- test_set %>%
  group_by(userId) %>%
  summarize(rating_count=n()) %>%
  summarise(descr="Rating counts for users",
            min=min(rating_count),max=max(rating_count),
            mean=mean(rating_count),median=median(rating_count))

user_stats
  
#Rating counts for users
test_set %>%
  group_by(userId) %>%
  summarize(rating_count=n()) %>%
  arrange(desc(rating_count)) %>%
  mutate(nr=row_number()) %>%
  ggplot(aes(nr,rating_count)) +
  geom_bar(stat="identity", col = "black") +
  geom_hline(aes(yintercept=as.numeric(user_stats[1,4]),color="red")) +
  guides(color = "none") +
  theme(axis.text.x = element_blank()) +
  xlab("users")+ylab("number of ratings")+
  ggtitle("Rating counts for users")

#Rating distribution for users histogram
test_set %>%
  group_by(userId) %>%
  summarize(avg_score=mean(rating)) %>%
  ggplot(aes(avg_score)) +
  geom_histogram(binwidth = 0.1, fill = "cyan", col = "black") +
  xlab("average rating")+ylab("number of users")+
  ggtitle("Ratings distribution for users")

#Rating counts for genres summary
genres_stats <- test_set %>%
  group_by(genres) %>%
  summarize(rating_count=n()) %>%
  summarise(descr="Rating counts for genres",
            min=min(rating_count),max=max(rating_count),
            mean=mean(rating_count),median=median(rating_count))

genres_stats
  
#Rating counts for genres
test_set %>%
  group_by(genres) %>%
  summarize(rating_count=n()) %>%
  arrange(desc(rating_count)) %>%
  mutate(nr=row_number()) %>%
  ggplot(aes(nr,rating_count)) +
  geom_bar(stat="identity", col = "black") +
  geom_hline(aes(yintercept=as.numeric(genres_stats[1,4]),color="red")) +
  guides(color = "none") +
  theme(axis.text.x = element_blank()) +
  xlab("genres")+ylab("number of ratings") +
  ggtitle("Rating counts for genres")

#Rating distribution for genres histogram
test_set %>%
  group_by(genres) %>%
  summarize(avg_score=mean(rating)) %>%
  ggplot(aes(avg_score)) +
  geom_histogram(binwidth = 0.1, fill = "cyan", col = "black") +
  xlab("average rating")+ylab("number of genres")+
  ggtitle("Ratings distribution for genres")


#Observations:
# - Mean rating looks good start of the model development
# - Movie,user and genres histogram follows similar distribution with mean at center (not a surprize)
# - Wide range of rating counts for movies,users and genres
# - Genres are group of movie types, some of them are much more common than others


##########################################################
# Model preparation
##########################################################

#Model : Median rating alone
median_rating = median(train_set$rating)
median_rating
Evaluate_Model(median_rating,"median alone")

#Model : Mean rating alone
mean_rating = mean(train_set$rating)
mean_rating
Evaluate_Model(mean_rating,"mean alone")
Evaluate_Model(Round_Rating(mean_rating),"mean rounded alone")

#User effect alone
user_effect_alone <- train_set %>%
  group_by(userId) %>%
  summarize(u_e = mean(rating-mean_rating),u_n=n())
head(user_effect_alone)

predicted_ratings <- test_set %>%
  left_join(user_effect_alone,by=c("userId")) %>%
  rowwise() %>%
  mutate(y=sum(mean_rating,u_e,na.rm = TRUE)) %>%
  pull(y)
Evaluate_Model(predicted_ratings,"user effect alone")
Evaluate_Model(Round_Rating(predicted_ratings),"user effect alone int")

#Movie effect alone
movie_effect_alone <- train_set %>%
  group_by(movieId) %>%
  summarize(m_e = mean(rating-mean_rating),m_n=n())
head(movie_effect_alone)

predicted_ratings <- test_set %>%
  left_join(movie_effect_alone,by=c("movieId")) %>%
  rowwise() %>%
  mutate(y=sum(mean_rating,m_e,na.rm = TRUE)) %>%
  pull(y)
Evaluate_Model(predicted_ratings,"movie effect alone")
Evaluate_Model(Round_Rating(predicted_ratings),"movie effect alone int")

#Genre effect alone
genre_effect_alone <- train_set %>%
  group_by(genres) %>%
  summarize(g_e = mean(rating-mean_rating),g_n=n())
head(genre_effect_alone)

predicted_ratings <- test_set %>%
  left_join(genre_effect_alone,by=c("genres")) %>%
  rowwise() %>%
  mutate(y=sum(mean_rating,g_e,na.rm = TRUE)) %>%
  pull(y)
Evaluate_Model(predicted_ratings,"genre effect alone")
Evaluate_Model(Round_Rating(predicted_ratings),"genre effect alone int")

#We see from the previous section that choosing the mean looks better than median.
#The order of possible effects will be chosen by RMSE decrease (larger decrease used first)


##########################################################
# Development of the model
# mean+movie+user+genre effect
##########################################################

#Mean rating
mean_rating = mean(train_set$rating)
mean_rating
e_mean <- Evaluate_Model(mean_rating,"mean")
e_mean

#Add movie effect:
movie_effect <- train_set %>%
  group_by(movieId) %>%
  summarize(m_e = mean(rating-mean_rating),m_n=n())
head(movie_effect)

predicted_ratings <- test_set %>%
  left_join(movie_effect,by=c("movieId")) %>%
  rowwise() %>%
  mutate(y=sum(mean_rating,m_e,na.rm = TRUE)) %>%
  pull(y)
e_mean_movie <- Evaluate_Model(predicted_ratings,"mean + movie effect")
e_mean_movie

#Add user effect:
user_effect <- train_set %>%
  left_join(movie_effect,by=c("movieId")) %>%
  group_by(userId) %>%
  summarize(u_e = mean(rating-mean_rating-m_e),u_n=n())
head(user_effect)

predicted_ratings <- test_set %>%
  left_join(movie_effect,by=c("movieId")) %>%
  left_join(user_effect,by=c("userId")) %>%
  rowwise() %>%
  mutate(y=sum(mean_rating,m_e,u_e,na.rm = TRUE)) %>%
  pull(y)
e_mean_movie_user <- Evaluate_Model(predicted_ratings,"mean + movie + user effect")
e_mean_movie_user

#Add genre effect:
genre_effect <- train_set %>%
  left_join(movie_effect,by=c("movieId")) %>%
  left_join(user_effect,by=c("userId")) %>%
  group_by(genres) %>%
  summarize(g_e = mean(rating-mean_rating-m_e-u_e),g_n=n())

predicted_ratings <- test_set %>%
  left_join(movie_effect,by=c("movieId")) %>%
  left_join(user_effect,by=c("userId")) %>%
  left_join(genre_effect,by=c("genres")) %>%
  rowwise() %>%
  mutate(y=sum(mean_rating,m_e,u_e,g_e,na.rm = TRUE)) %>%
  pull(y)
e_mean_movie_user_genre <- Evaluate_Model(predicted_ratings,"mean + movie + user effect")
e_mean_movie_user_genre

##########################################################
# Model optimization with lambda penality
# mean+movie+user+genre effect+lambda
##########################################################

#Function to calculate RMSE for a set of lambdas
Calculate_RMSEs_for_Lambda <- function(l) {
  rmses_lambda <- sapply(l, function(l){
    mean_rating = mean(train_set$rating)
    movie_effect <- train_set %>%
      group_by(movieId) %>%
      summarize(m_e = sum(rating-mean_rating)/(n()+l))
    user_effect <- train_set %>%
      left_join(movie_effect,by=c("movieId")) %>%
      group_by(userId) %>%
      summarize(u_e = sum(rating-mean_rating-m_e)/(n()+l))
    genre_effect <- train_set %>%
      left_join(movie_effect,by=c("movieId")) %>%
      left_join(user_effect,by=c("userId")) %>%
      group_by(genres) %>%
      summarize(g_e = sum(rating-mean_rating-m_e-u_e)/(n()+l))
    predicted_ratings <- test_set %>%
      left_join(movie_effect,by=c("movieId")) %>%
      left_join(user_effect,by=c("userId")) %>%
      left_join(genre_effect,by=c("genres")) %>%
      rowwise() %>%
      mutate(y=sum(mean_rating,m_e,u_e,g_e,na.rm = TRUE)) %>%
      pull(y)
    return(RMSE(predicted_ratings, test_set$rating))
  })
}

#Try lambdas from 1 to 10
lambdas <- seq(0:10)
rmse_list <- Calculate_RMSEs_for_Lambda(lambdas)
qplot(lambdas,rmse_list)

#Finetune the lambda from 4 to 6
lambdas <- seq(4,6,0.25)
rmse_list <- Calculate_RMSEs_for_Lambda(lambdas)
qplot(lambdas,rmse_list)

# Model with the selected lamba penality (4.755)
lambda <- 4.755
mean_rating = mean(train_set$rating)
movie_effect <- train_set %>%
  group_by(movieId) %>%
  summarize(m_e = sum(rating-mean_rating)/(n()+lambda))
user_effect <- train_set %>%
  left_join(movie_effect,by=c("movieId")) %>%
  group_by(userId) %>%
  summarize(u_e = sum(rating-mean_rating-m_e)/(n()+lambda))
genre_effect <- train_set %>%
  left_join(movie_effect,by=c("movieId")) %>%
  left_join(user_effect,by=c("userId")) %>%
  group_by(genres) %>%
  summarize(g_e = sum(rating-mean_rating-m_e-u_e)/(n()+lambda))
predicted_ratings <- test_set %>%
  left_join(movie_effect,by=c("movieId")) %>%
  left_join(user_effect,by=c("userId")) %>%
  left_join(genre_effect,by=c("genres")) %>%
  rowwise() %>%
  mutate(y=sum(mean_rating,m_e,u_e,g_e,na.rm = TRUE)) %>%
  pull(y)
Evaluate_Model(predicted_ratings,"mean + movie + user effect + lambda")

#Check the prediction range for our Model 8
data.frame(Description="Rating range after prediction",
           min=min(predicted_ratings),max=max(predicted_ratings),
           mean=round(mean(predicted_ratings),2),median=median(predicted_ratings))

#As we see we can cut range to fit better for [0.5,5]

# Final model with mean+movie+user+genres+lambda+interval_cut
lambda <- 4.755
mean_rating = mean(train_set$rating)
movie_effect <- train_set %>%
  group_by(movieId) %>%
  summarize(m_e = sum(rating-mean_rating)/(n()+lambda))
user_effect <- train_set %>%
  left_join(movie_effect,by=c("movieId")) %>%
  group_by(userId) %>%
  summarize(u_e = sum(rating-mean_rating-m_e)/(n()+lambda))
genre_effect <- train_set %>%
  left_join(movie_effect,by=c("movieId")) %>%
  left_join(user_effect,by=c("userId")) %>%
  group_by(genres) %>%
  summarize(g_e = sum(rating-mean_rating-m_e-u_e)/(n()+lambda))
predicted_ratings <- test_set %>%
  left_join(movie_effect,by=c("movieId")) %>%
  left_join(user_effect,by=c("userId")) %>%
  left_join(genre_effect,by=c("genres")) %>%
  rowwise() %>%
  mutate(y=sum(mean_rating,m_e,u_e,g_e,na.rm = TRUE)) %>%
  mutate(y=ifelse(y>5,5,y),y=ifelse(y<0.5,0.5,y)) %>%
  pull(y)
Evaluate_Model(predicted_ratings,"mean + movie + user + genres + lambda + int")

#0.8636913

##########################################################
# Final model evaluation
# Model : Mean + movie + user + genre effect + lambda + interval_cut
# Dataset = final_holdout_test (only used here)
##########################################################

predicted_ratings <- final_holdout_test %>%
  left_join(movie_effect,by=c("movieId")) %>%
  left_join(user_effect,by=c("userId")) %>%
  left_join(genre_effect,by=c("genres")) %>%
  rowwise() %>%
  mutate(y=sum(mean_rating,m_e,u_e,g_e,na.rm = TRUE)) %>%
  mutate(y=ifelse(y>5,5,y),y=ifelse(y<0.5,0.5,y)) %>%
  pull(y)
Evaluate_Model(predicted_ratings,"mean + movie + user effect + lambda + int",y = final_holdout_test$rating)

#Evaluate by caret::RMSE function (objective : RMSE < 0.86490)
caret::RMSE(predicted_ratings,final_holdout_test$rating)

#

#Later possibilities for improvement:
# - Split genres into individuals like (Action, Romance,etc...) and calculate effects for each
# - As 'Ratings in general' historgram shows users are more likely to give integer values. Maybe can use this information to refine the model
# - Clustering users and movies by similarities can also be an option because this way we minimize missing values int the User/Movie matrix



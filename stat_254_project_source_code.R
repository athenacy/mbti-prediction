library(tidyverse)
library(xtable)

# read in data from CSV file
raw_dat = read.csv("mbti_1.csv")

# convert each MBTI type to an integer factor based on its extraverted function
convert_to_factor = function(type) {
  convert_one = function(t) {
    return(switch(t, 
                  ENTP=, ENFP=, INTP=, INFP = 0,  # Ne
                  ISTP=, ESTP=, ESFP=, ISFP = 1,  # Se
                  INTJ=, ENTJ=, ESTJ=, ISTJ = 2,  # Te
                  INFJ=, ENFJ=, ESFJ=, ISFJ = 3)) # Fe
  }
  return(as.factor(sapply(type, convert_one)))
}

dat = mutate(raw_dat, type = convert_to_factor(type))

counts_per_type = dat %>% group_by(type) %>% count()

# for some users, the text in the "posts" column is surrounded by single quotes. 
# we need to remove them
remove_quotes = function(stri) {
  if (str_sub(stri[1], 1, 1) == "'") {
    return(str_sub(stri, 2, -2))
  }
  else {
    return(stri)
  }
}

dat$posts = sapply(dat$posts, remove_quotes)

# generate indices corresponding to the rows that are used for the training set
# this allows us to easily obtain the training data from any version of the 
# transformed dataset
set.seed(1234)
N = nrow(dat)
train = sample(1:N, round(N*0.8))

# add column containing number of links per corpus
dat_links = mutate(dat, link = as.numeric(str_count(posts, "http[s]?:\\/\\/"))) %>%
  group_by(type)

total_count = count(dat_links[train,])
link_count = summarize(dat_links[train,], links = sum(link))
links_per_type = mutate(link_count, avg_links = links/total_count$n)[,-2]
links_per_type

# remove links from each corpus
no_links = mutate(dat_links, posts = str_remove_all(posts, "http[s]?:\\/\\/[^\\| ]*\\b"))

# in each corpus, each post is separated by three pipe characters
# we split each corpus in each row into a vector
# with each element containing a post
split_posts = mutate(no_links, posts = str_split(posts, "\\|\\|\\|"))

# remove empty elements in resulting post vectors
remove_empty = function(stri) {
  return(stri[nzchar(stri)])
}

split_posts$posts = lapply(split_posts$posts, remove_empty)

# download UDPipe model
library(udpipe)
dl = udpipe_download_model(language = "english-ewt")
udmodel = udpipe_load_model(file = dl$file_model)

# for each person, annotate their posts
# and store the resulting data frames in a new column
get_words = function(posts) {
  words = udpipe_annotate(udmodel, posts)
  words_df = as.data.frame(words)
  return(words_df)
}
 
dat_words = mutate(split_posts, words = lapply(posts, get_words))
dat_words = dat_words[,-2]

find_adj_ratio = function(words) {
  word_count = words %>% group_by(doc_id) %>% count()
  adj_count = words %>% group_by(doc_id) %>% summarise(n = sum(upos == "ADJ" | upos == "ADV"))

  return(adj_count$n / word_count$n)
}

dat_adj = mutate(dat_words, adj_ratio = lapply(words, find_adj_ratio))

dat_adj$mean_adj_ratio = sapply(dat_adj$adj_ratio, mean)

adj_by_group = dat_adj[train,] %>% 
  mutate(num_posts = sapply(words, function(x) length(unique(x$doc_id))), weighted_adj_ratio = mean_adj_ratio * num_posts) %>% 
  group_by(type)
adj_ratio_per_type = summarize(adj_by_group, adj_ratio = sum(weighted_adj_ratio)/sum(num_posts))

data_by_type = split(dat_words[train,], dat_words[train,]$type)
words_by_type = lapply(data_by_type, function(x) group_by(bind_rows(x$words), lemma))
word_count = lapply(words_by_type, count)

word_count_ratios = lapply(word_count, function(x) data.frame(lemma = x$lemma, n = x$n/sum(x$n)))

library(purrr)
word_count_all = reduce(word_count_ratios, merge, by = "lemma")
colnames(word_count_all) = c("lemma", "n0", "n1", "n2", "n3")

# sort dataframe by most to least frequently used words
word_count_sorted = word_count_all[order(word_count_all$n0 + 
                                           word_count_all$n1 + 
                                           word_count_all$n2 + 
                                           word_count_all$n3, decreasing = TRUE),]

word_count_filtered = filter(word_count_sorted, (pmax(n0, n1, n2, n3) - pmin(n0, n1, n2, n3))/pmin(n0, n1, n2, n3) > 0.5)
keywords_df = word_count_filtered[1:200,]
keywords = word_count_filtered$lemma[1:200]

# remove overfitting keywords
keywords_new = str_subset(keywords, "([IiEe][NnSs][FfTt][PpJj][sS]?|^[NSTFnstf][ei]$|Mbto|PerC|Tapatalk|^[SsNn][FfTt]$|^[is]$|\\[|7|^[so]p$)", negate = TRUE)

find_num_keywords = function(words, keywords) {
  counts = words$lemma %>% table()
  num_keywords = counts[match(keywords, names(counts))]
  num_keywords[is.na(num_keywords)] = 0
  return(as.vector(num_keywords))
}

num_keywords = lapply(dat_words$words, find_num_keywords, keywords = keywords_new)
num_keywords_df = as.data.frame(do.call(rbind, num_keywords))

colnames(num_keywords_df) = keywords_new
dat_final = cbind(dat_words[,1:2], num_keywords_df)

# remove/replace special characters in column names 
# to work with model fitting functions
library(janitor)
dat_final_cn = clean_names(dat_final)

# split final dataset into train and test data
train_dat = dat_final_cn[train,]
test_dat = dat_final_cn[-train,]

library(nnet)
logi_fit = multinom(type ~ ., train_dat)
pred = predict(logi_fit, test_dat, type = "class")
# compare predicted classes to actual classes for test data
mean(test_dat$type == pred)

library(glmnetUtils)
# use cross-validation to tune penalty term lambda
cv = cv.glmnet(type ~ ., train_dat, family = "multinomial", type.multinomial = "grouped")

laslogi_fit = glmnet(type ~ ., train_dat, family = "multinomial", lambda = cv$lambda.min, type.multinomial = "grouped")
pred = predict(laslogi_fit, test_dat, type = "class")
# compare predicted classes to actual classes for test data
mean(test_dat$type == pred)

beta = coef(laslogi_fit)$`0`
beta_df = data.frame(coefficient = beta[,1])
rownames(beta_df) = c("Intercept", "link", keywords_new)
smallest_coefs = slice(beta_df, order(abs(beta_df$coefficient))) %>% head(10)

library(randomForest)
set.seed(1234)
rf_model = randomForest(type ~ ., train_dat, importance = TRUE)
rf_pred = predict(rf_model, test_dat)
# compare predicted classes to actual classes for test data
mean(rf_pred == test_dat$type)

# get indices of the variables with coefficients of 0 in the penalized model
zero_coefs = order(abs(beta_df$coefficient))[1:5]
rf_model_reduced = randomForest(type ~ ., train_dat[,-zero_coefs], importance = TRUE)
rf_pred_reduced = predict(rf_model_reduced, test_dat[,-zero_coefs])
mean(rf_pred_reduced == test_dat[,-zero_coefs]$type)

type_counts = raw_dat %>% group_by(type) %>% count()
type_counts$n = type_counts$n / sum(type_counts$n) * 100
colnames(type_counts)[2] = "data_percentage"
type_counts2 = cbind(type_counts, pop_percentage = c(2.5, 8.1, 1.8, 3.2, 12.3, 8.5, 8.7, 4.3, 1.5, 4.4, 2.1, 3.3, 13.8, 8.8, 11.16, 5.4))

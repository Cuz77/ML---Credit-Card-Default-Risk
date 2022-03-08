#---------------------------------------------------------------------------------------#
#                                                                                       #
#                            INTRODUCTION AND OBJECTIVES                                #
#                                                                                       #
#---------------------------------------------------------------------------------------#

#----------------------------------------------->
# This project originates from the kaggle's data set "Credit Card Approval Prediction" that can be found under this link: 
# https://www.kaggle.com/rikdifos/credit-card-approval-prediction. The data is saved in the credit_data.rda file and 
# contains credit card holders data with corresponding credit records containing personal information along with the status 
# of the credit (e.g. whether the customer defaults and for how long). 
# 
# The goal of this project is to predict whether the applicant is going to be at risk of default based on provided features.
# Due to economic hardships, the bank wants to introduce an additional vetting step to potentially limit credit lines for 
# customers who may be at risk of default in the future.
# Later, we will define what constitutes a risky customer for the purpose of this project.
#
# The project files can be found under the following address: https://github.com/Cuz77/ML-Credit-Card-Default-Risk
#----------------------------------------------->

#---------------------------------------------------------------------------------------#
#                                                                                       #
#                                PREPARE THE DATA SET                                   #
#                                                                                       #
#---------------------------------------------------------------------------------------#

#----------------------------------------------->
# Load and install libraries if applicable
#----------------------------------------------->

if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(smotefamily)) install.packages("smotefamily", repos = "http://cran.us.r-project.org")
if(!require(this.path)) install.packages("this.path", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")

library(caret)
library(gridExtra)
library(kableExtra)
library(randomForest)
library(smotefamily)
library(this.path)
library(tidyverse)

setwd(this.path::this.dir())       # set the working directory to this file's folder
load(file="credit_data.rda")       # load the data set

#---------------------------------------------------------------------------------------#
#                                                                                       #
#                                 DATA SET OVERVIEW                                     #
#                                                                                       #
#---------------------------------------------------------------------------------------#

#----------------------------------->
# First, analyze basic data set characteristics 
#----------------------------------->


# 771,715 rows and 20 columns
dim(data)

# summarize the data set
t(summary(data)[c(1,3,4,6),])

# 36,457 unique customers
length(unique(data$ID))

# There is no NA values in the data set
sum(is.na(data))

# Most credit lines on record are between 3 and 14 months old
data %>% group_by(ID) %>%
  summarize(months=n()) %>%
  ggplot(aes(months)) +
  geom_bar(aes(fill=months>2&months<14)) +
  scale_x_continuous(breaks=seq(1,60,2)) +
  scale_fill_manual(values = c('red', 'black') ) +
  theme(legend.position="none")


#---------------------------------------------------------------------------------------#
#                                                                                       #
#                                 BUSINESS OBJECTIVE                                    #
#                                                                                       #
#---------------------------------------------------------------------------------------#

#----------------------------------->
# The data set currently has multiple entries per applicant. The statuses are as follows:
# 0: 1-29 days past due
# 1: 30-59 days past due
# 2: 60-89 days overdue
# 3: 90-119 days overdue
# 4: 120-149 days overdue
# 5: Overdue or bad debts, write-offs for more than 150 days
# C: paid off that month
# X: No loan for the month
#
# As presented earlier, the data set has 771,715 entries for 36,457 unique customers. We do not need all of them since features are identical 
# across each customer's history and keeping different numbers of duplicates for customers would distort the data (i.e. we would count each 
# feature x 16 times where x is number of months on record). We will therefore group the data set by applicant ID. First, however, we need 
# to construct some measure of a good or bad customer.
#----------------------------------->


cnames <- colnames(data)[-c(2,3)]                         # store column names except for MONTHS and STATUS
data %>% group_by(.dots=cnames) %>% summarize() %>%       # this proves that features do not change across months
  group_by(ID) %>% summarize(n=n()) %>% filter(n!=1)      # (except for status and month)


#----------------------------------->
# For the purpose of this project, a risky customer is one that has defaulted for over 30 days at any time (indicating some financial problems).
#----------------------------------->

scoring_data <- data %>%
  group_by(.dots=cnames) %>%
  summarize(
    RISKY = case_when(            # over 30 days due - mark as risky
      any(STATUS == 2 |           
            STATUS == 3 |         
            STATUS == 4 |         
            STATUS == 5) ~ 1,     
      TRUE ~ 0                    # otherwise mark as not risky
    )
  ) %>% as.data.frame()


#----------------------------------->
# There are only about 1.69% risky customers in the data set showing how imbalanced it is.
#----------------------------------->


round(mean(scoring_data$RISKY==1) * 100,2)                # [1.69]


#---------------------------------------------------------------------------------------#
#                                                                                       #
#                                  DATA SET ANALYSIS                                    #
#                                                                                       #
#---------------------------------------------------------------------------------------#

#----------------------------------->
# Investigate features, starting with categorical ones.
#----------------------------------->


# str
str(scoring_data)

# shorter names for clarity
colnames(scoring_data)[colnames(scoring_data) == "OCCUPATION_TYPE"] <- "OCCUPATION"
colnames(scoring_data)[colnames(scoring_data) == "NAME_INCOME_TYPE"] <- "INCOME_SRC"
colnames(scoring_data)[colnames(scoring_data) == "NAME_EDUCATION_TYPE"] <- "EDUCATION"
colnames(scoring_data)[colnames(scoring_data) == "NAME_FAMILY_STATUS"] <- "FAMILY"
colnames(scoring_data)[colnames(scoring_data) == "NAME_HOUSING_TYPE"] <- "HOUSING"

# Binary features analysis
plot <- scoring_data %>%
  group_by(RISKY) %>%
              # Transform characters into boolean values and calculate mean
  summarize(car=mean(FLAG_OWN_CAR=="Y"),
            realty=mean(FLAG_OWN_REALTY=="Y"),
            male=mean(CODE_GENDER=="M"),
            mobile=mean(FLAG_MOBIL==1),
            work_phone=mean(FLAG_WORK_PHONE==1),
            email=mean(FLAG_EMAIL==1),
            phone=mean(FLAG_PHONE==1)
  ) %>%       
  pivot_longer(2:8, "cat") %>%              # Transform for easier plotting
  ggplot(aes(RISKY, value, fill=cat)) +
  geom_bar(stat="identity") +
  facet_wrap(cat~., nrow=4) +
  theme_bw() +
  theme(legend.position="none") +
  scale_x_continuous(breaks=c(0,1)) +
  labs(y="PROPORTION")

# Need to keep groups with zero instances for all categories, hence cannot use simple grouping
categories <- c("INCOME_SRC", "EDUCATION", "FAMILY", "HOUSING", "OCCUPATION")
values <- scoring_data %>% select(categories) %>% pivot_longer(categories) %>% group_by(name, value) %>% summarize()

# All groups with count of instances
right_ordinal <- scoring_data %>% filter(RISKY==1) %>% 
  select(categories) %>% pivot_longer(categories) %>% group_by(name, value) %>%
  summarize(n=n()) 

# Join all instances with their counts and replace those with NA counts with 0s
ordinal_dataset <- values %>% left_join(right_ordinal, by=c("name", "value")) %>% mutate_if(is.numeric, ~replace(., is.na(.), 0))


#----------------------------------->
# Plot the categorical features in a 2x2 grid.
#----------------------------------->


# Income source plot
p1 <- ordinal_dataset %>%
  filter(name=="INCOME_SRC") %>%
  ggplot(aes(value, n)) +
  theme_bw() +
  geom_bar(stat="identity", fill="#3b3f40") +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=0.5),
        legend.position="none") +
  labs(x="", y="")

# Education type plot
p2 <- ordinal_dataset %>%
  filter(name=="EDUCATION") %>%
  ggplot(aes(value, n)) +
  theme_bw() +
  geom_bar(stat="identity", fill="#3b3f40") +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=0.5),
        legend.position="none") +
  labs(x="", y="")

# Marital status plot
p3 <- ordinal_dataset %>%
  filter(name=="FAMILY") %>%
  ggplot(aes(value, n)) +
  geom_bar(stat="identity", fill="#3b3f40") +
  theme_bw() +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=0.5),
        legend.position="none") +
  labs(x="", y="")

# Housing conditions plot
p4 <- ordinal_dataset %>%
  filter(name=="HOUSING") %>%
  ggplot(aes(value, n)) +
  geom_bar(stat="identity", fill="#3b3f40") +
  theme_bw() +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=0.5),
        legend.position="none") +
  labs(x="", y="")

grid.arrange(p1,p2,p3,p4)           # plot above 2 charts in a 2x2 grid


#----------------------------------->
# Need to compare population proportions among different categories of features (e.g. the percentage of married people among 
# risky customers and in the overall population).
# The following code calculates categories' proportions in the overall population and among 
# risky customers and subtract them. Repeat 5 times - once per each category.
#----------------------------------->


df <- data.frame(table(scoring_data$EDUCATION))         # group data by education type
colnames(df) <- c("value", "n2")
edu_df <- ordinal_dataset %>% filter(name=="EDUCATION") %>% left_join(df, "value") %>% 
  group_by(name) %>% 
  mutate(alln=sum(n), all2=sum(n2)) %>%
  ungroup() %>% mutate(freq_risky=n/alln, freq_all=n2/all2, difference = freq_risky - freq_all) %>%
  select(name, value, difference)

df <- data.frame(table(scoring_data$FAMILY))            # group data by marital status
colnames(df) <- c("value", "n2")
fam_df <- ordinal_dataset %>% filter(name=="FAMILY") %>% left_join(df, "value") %>% 
  group_by(name) %>% 
  mutate(alln=sum(n), all2=sum(n2)) %>%
  ungroup() %>% mutate(freq_risky=n/alln, freq_all=n2/all2, difference = freq_risky - freq_all) %>%
  select(name, value, difference)

df <- data.frame(table(scoring_data$HOUSING))           # group data by housing conditions
colnames(df) <- c("value", "n2")
house_df <- ordinal_dataset %>% filter(name=="HOUSING") %>% left_join(df, "value") %>% 
  group_by(name) %>% 
  mutate(alln=sum(n), all2=sum(n2)) %>%
  ungroup() %>% mutate(freq_risky=n/alln, freq_all=n2/all2, difference = freq_risky - freq_all) %>%
  select(name, value, difference)

df <- data.frame(table(scoring_data$INCOME_SRC))        # group data by income source
colnames(df) <- c("value", "n2")
inc_df <- ordinal_dataset %>% filter(name=="INCOME_SRC") %>% left_join(df, "value") %>% 
  group_by(name) %>% 
  mutate(alln=sum(n), all2=sum(n2)) %>%
  ungroup() %>% mutate(freq_risky=n/alln, freq_all=n2/all2, difference = freq_risky - freq_all) %>%
  select(name, value, difference)

# combine all of the above in one table and sort by the highest difference
bind_rows(list(edu_df, fam_df, house_df, inc_df)) %>% arrange(desc(abs(difference)))


#----------------------------------->
# Lots of people have no specific occupation on record.
#----------------------------------->

# Occupation plot
p5 <- ordinal_dataset %>%
  filter(name=="OCCUPATION") %>%
  ggplot(aes(value, n)) +
  geom_bar(stat="identity", fill="#3b3f40") +
  theme_bw() +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=0.5),
        legend.position="none") +
  labs(x="", y="")

# replace empty occupation with a "None" for later use
scoring_data <- scoring_data %>% mutate(OCCUPATION=if_else(OCCUPATION=="", "None", OCCUPATION))


#----------------------------------->
# Density plots for work tenure, income, and age with a bar plot for employment status
#----------------------------------->

# Density plot for income
income_plot <- scoring_data %>%
  ggplot(aes(AMT_INCOME_TOTAL, fill=as.factor(RISKY))) + 
  geom_density(bw=15000, alpha=0.3) +                 # smooth the line a bit(bw=15000) to reflect trends
  scale_y_continuous(labels = scales::comma) +
  theme_bw() +
  scale_fill_manual(values=c("#53d1ee", "#fed20c")) +
  scale_x_continuous(limits=c(0,500000), labels = scales::comma) + 
  theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=0.5), legend.position="none") +
  labs(y="", x="income")

# Transform the time interval features
scoring_data <- scoring_data %>% 
  mutate(years_old = DAYS_BIRTH/365*-1,                   # convert from negative # of days to positive # of years
         years_employed = case_when(                      # similar as above - positive number (always 365243) is for unemployed
           DAYS_EMPLOYED < 0 ~ DAYS_EMPLOYED/365*-1,
           TRUE ~ 0),
         employed = if_else(years_employed == 0, 0, 1))   # create a binary feature - unemployed vs employed

# Density plot for age
age_plot <- scoring_data %>%
  ggplot(aes(years_old, fill=as.factor(RISKY))) +
  geom_density(alpha=0.3) +
  theme_bw() +
  theme(legend.position="none") +
  labs(y="", x="age") +
  annotate("rect", xmin=28.7, xmax=34, ymin=0, ymax=0.0305, alpha=0.1, fill="#3b3f40") +
  annotate("rect", xmin=28.7, xmax=47.5, ymin=0, ymax=0.0305, alpha=0.1, fill="#3b3f40") +
  annotate("rect", xmin=28.7, xmax=57.5, ymin=0, ymax=0.0305, alpha=0.1, fill="#3b3f40") +
  scale_fill_manual(values=c("#53d1ee", "#fed20c"))

# Density plot for work tenure 
employement_time_plot <- scoring_data %>%
  ggplot(aes(years_employed, fill=as.factor(RISKY))) +
  geom_density(alpha=0.3) +
  theme_bw() +
  annotate("rect", xmin=0.9, xmax=3.8, ymin=0, ymax=0.15, alpha=0.2, fill="#3b3f40") +
  scale_fill_manual(values=c("#53d1ee", "#fed20c")) +
  theme(legend.position="none") +
  labs(y="", x="tenure")

# Bar plot for employment status
employment_plot <- scoring_data %>%
  group_by(RISKY) %>%
  summarize(employed=mean(employed=="0")) %>%
  ggplot(aes(RISKY, employed, fill=RISKY)) +
  geom_bar(stat="identity", fill=c("#53d1ee", "#fed20c")) +
  theme_bw() +
  labs(y="unemployment percentage", x="") +
  theme(legend.position="none") +
  scale_x_continuous(breaks=c(0,1), labels=c("NOT-RISKY", "RISKY"))

grid.arrange(income_plot, age_plot, employement_time_plot, employment_plot)   # plot above 2 charts in a 2x2 grid


#----------------------------------->
# Boxplots for family size and number of children
#----------------------------------->


children_plot <- scoring_data %>%
  ggplot(aes(RISKY, CNT_CHILDREN, fill=as.factor(RISKY))) +
  scale_y_continuous(limits=c(0, 20)) +
  geom_boxplot() +
  theme_bw() +
  theme(legend.position="none") +
  scale_fill_manual(values=c("#53d1ee", "#fed20c")) +
  labs(y="", x="") +
  scale_x_continuous(breaks=c(0,1), labels=c("NOT_RISKY", "RISKY"))


family_plot <- scoring_data %>%
  ggplot(aes(RISKY, CNT_FAM_MEMBERS, fill=as.factor(RISKY))) +
  scale_y_continuous(limits=c(0, 20)) +
  geom_boxplot() +
  theme_bw() +
  theme(legend.position="none") +
  scale_fill_manual(values=c("#53d1ee", "#fed20c")) +
  labs(y="", x="") +
  scale_x_continuous(breaks=c(0,1), labels=c("NOT_RISKY", "RISKY"))

grid.arrange(children_plot, family_plot, nrow=1)  # plot above 2 charts in one row


#---------------------------------------------------------------------------------------#
#                                                                                       #
#                                  DATA PREPARATION                                     #
#                                                                                       #
#---------------------------------------------------------------------------------------#

#----------------------------------->
# Transform binary categories denoted with characters to numerical
#----------------------------------->


scoring_data <- scoring_data %>% 
  mutate(CODE_GENDER = if_else(CODE_GENDER == "M", 1, 0),
         FLAG_OWN_CAR = if_else(FLAG_OWN_CAR == "Y", 1, 0),
         FLAG_OWN_REALTY = if_else(FLAG_OWN_REALTY == "Y", 1, 0)
         )


#----------------------------------->
# Perform one-hot encoding on categorical features
#----------------------------------->


dummy <- dummyVars(" ~ FAMILY + EDUCATION + HOUSING + OCCUPATION + INCOME_SRC", 
                   data=scoring_data)
m <- data.frame(predict(dummy, newdata=scoring_data))           # create binary columns               
clean_dataset <- bind_cols(m, scoring_data)                     # join columns

# get rid of unnecessary columns
clean_dataset <- clean_dataset %>% 
  select(-c(
                   # remove categorical features
    "INCOME_SRC", "EDUCATION", 
    "FAMILY", "HOUSING", 
    "OCCUPATION", "ID",
    "DAYS_BIRTH", "DAYS_EMPLOYED",
    "CNT_CHILDREN", "CNT_FAM_MEMBERS",
                   # remove binary feature that do not add any value
    "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "CODE_GENDER", "FLAG_MOBIL", "FLAG_WORK_PHONE",
    "FLAG_PHONE", "FLAG_EMAIL", "employed"
  ))                


#----------------------------------->
# Normalize the data set high value features (income, age, and work tenure) with a min-max function.
#----------------------------------->
  

clean_dataset <- clean_dataset %>%
  mutate(AMT_INCOME_TOTAL = (AMT_INCOME_TOTAL - mean(AMT_INCOME_TOTAL)) / sd(AMT_INCOME_TOTAL),
         years_old = (years_old - mean(years_old)) / sd(years_old),
         years_employed = (years_employed - mean(years_employed)) / sd(years_employed))


#----------------------------------->
# Split the data set between three sets (train, test, and validation -> 90-10-10)
#----------------------------------->


set.seed(777, sample.kind="Rounding")                                        # ensure reproducibility
train_index <- createDataPartition(clean_dataset$RISKY, p=0.8, list=FALSE)   # construct train index for 80% of observations
temp_set <- slice(clean_dataset, -train_index)                               # temporary set to split in half
test_index <- createDataPartition(temp_set$RISKY, p=0.5, list=FALSE)         # construct 10% test index

train_set <- slice(clean_dataset, train_index)                               # 80% train data set
test_set <- slice(temp_set, test_index)                                      # 10% test set
validation_set <- slice(temp_set, -test_index)                               # 10% validation set


#----------------------------------->
#  Models evaluation will be done with Balanced Accuracy, F1, and Specificity with the confusionMatrix() function.
#----------------------------------->

#---------------------------------------------------------------------------------------#
#                                                                                       #
#                                   LINEAR MODEL                                        #
#                                                                                       #
#---------------------------------------------------------------------------------------#

#----------------------------------->
# Fit the Linear Model.
#----------------------------------->


fit_glm <- train(RISKY~.,                   # train linear model
                 data=train_set, 
                 method="glm")


#----------------------------------->
# Tune the cutoff to maximize F1, Specificity, and Balanced Accuracy
#----------------------------------->


tune <- seq(0.007, 0.1, by=0.005)           # define tuning parameters
pred_glm <- predict(fit_glm, train_set)     # create predictions for the train set

# apply different cutoff values to predictions to find the best one
tune_glm <- sapply(tune, function(tune){
  cm <- confusionMatrix(as.factor(if_else(pred_glm >= tune, 1, 0)),
                        as.factor(train_set$RISKY),
                        mode="everything")
  cm$byClass[c("Specificity", "F1", "Balanced Accuracy")]
}) %>% as.data.frame()

colnames(tune_glm) <- tune                  # set col names to parameter values
tune_plot <- data.frame(t(tune_glm))        # transpose for plotting
tune_plot$cutoff <- rownames(tune_plot)     # set row names to metric names

# plot tuning parameters to visually find the best one
tune_plot %>%
  ggplot(aes(y=Specificity, x=cutoff)) +
  geom_line(aes(group = 1)) +
  geom_line(aes(y=F1, group = 1), color="red") +
  geom_line(aes(y=Balanced.Accuracy, group = 1), color="purple") +
  theme_bw() +
  labs(y="outcome", x="cutoff value") +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=0.5))


#----------------------------------->
# Test the linear model against the test set with a cut-off being 0.017
#----------------------------------->


pred_glm <- predict(fit_glm, test_set)      # create predictions for the test set
cm_glm <- confusionMatrix(as.factor(if_else(pred_glm > 0.017, 1, 0)),
                          as.factor(test_set$RISKY),
                          mode="everything")

cm_glm$byClass[c("Specificity", "F1", "Balanced Accuracy")]
cm_glm$table


#---------------------------------------------------------------------------------------#
#                                                                                       #
#                                  DECISION TREE                                        #
#                                                                                       #
#---------------------------------------------------------------------------------------#

#----------------------------------->
# Define weights so that the model doesn't predict the majority class for all cases.
# The weight is 50 since the minority class makes up for under 2% of the data set
#----------------------------------->


loss_mtx <- matrix(c(0,50,1,0), 2)            


#----------------------------------->
# The weight matrix
#     [,1] [,2]
#[1,]   0    1    POSITIVE
#[2,]   50   0    NEGATIVE
#       T    F
#----------------------------------->


# fit the model
fit_tree <- train(as.factor(RISKY)~., 
                  data=train_set, 
                  method="rpart",
                  parms=list(loss=loss_mtx)     # apply weights
)

# evaluate against the test set
cm_tree <- confusionMatrix(data=predict(fit_tree, test_set), 
                           reference=as.factor(test_set$RISKY),
                           mode="everything")

# show relevant performance metrics
cm_tree$byClass[c("Specificity", "F1", "Balanced Accuracy")]
cm_tree$table


#---------------------------------------------------------------------------------------#
#                                                                                       #
#                                     RANDOM FOREST                                     #
#                                                                                       #
#---------------------------------------------------------------------------------------#

#----------------------------------->
# Fit the random forest with multiple decision trees. the ntree parameter is limited to 5 
# to decrease time needed for fitting the model.
#----------------------------------->


# fit random forest
fit_rf <- train(as.factor(RISKY)~.,
                data=train_set,
                method="rf",
                classwt=c(1,50),     # apply the same weights as for Decision Tree
                ntree=5)

cm_rf <- confusionMatrix(data=predict(fit_rf, test_set),
                         reference=as.factor(test_set$RISKY),
                         mode="everything")

cm_rf$byClass[c("Specificity", "F1", "Balanced Accuracy")]
cm_rf$table


#----------------------------------->
# Apply an oversampling technique to mitigate the imbalance factor. Weights will be retained due to importance of predicting
# positive outcome (risky customer), but with smaller values since the overfitting technique is already mitigating this issue.
#----------------------------------->


# over-sample the minority class
smote <- SMOTE(train_set, train_set$RISKY)
oversampled_set <- smote$data %>% select(-class)

# fit the random forest model again with the updated data set
fit_rf <- train(as.factor(RISKY)~.,
                data=oversampled_set,
                # keep the slightly lower weights because we want 
                # to identify risky customers above else
                classwt=c(1,15),    
                ntree=5,      # limit number of trees to decrease the run time
                method="rf"
)

# evaluate against the test set
smote_rf_cm <- confusionMatrix(data=predict(fit_rf, test_set),
                               reference=as.factor(test_set$RISKY),
                               mode="everything")

smote_rf_cm$byClass[c("Specificity", "F1", "Balanced Accuracy")]
smote_rf_cm$table


#---------------------------------------------------------------------------------------#
#                                                                                       #
#                         RESULTS AND FINAL VALIDATION                                  #
#                                                                                       #
#---------------------------------------------------------------------------------------#

#----------------------------------->
# First, print the results obtained so far.
#----------------------------------->


# bind all models' evaluations from previous sections
outcome <- bind_rows(smote_rf_cm$byClass[c("Specificity", "F1", "Balanced Accuracy")],
                     cm_rf$byClass[c("Specificity", "F1", "Balanced Accuracy")],
                     cm_tree$byClass[c("Specificity", "F1", "Balanced Accuracy")],
                     cm_glm$byClass[c("Specificity", "F1", "Balanced Accuracy")]) %>% data.frame()

# set row names for clarity
rownames(outcome) <- c("Random Forest with oversampling", "Random Forest with weights",
                       "Decision Tree", "Linear Model")

outcome


#----------------------------------->
# The best model appears to be the Random Forest with oversampled data set and decreased weights.
# Evaluate it against the validation set to get the final result.
#----------------------------------->


# evaluate against the validation set
final_cm <- confusionMatrix(data=predict(fit_rf, validation_set),
                            reference=as.factor(validation_set$RISKY),
                            mode="everything")
final_cm$byClass[c("Specificity", "F1", "Balanced Accuracy")]


#----------------------------------->
# Models' outcomes should reflect the following values:
# 
#                                 Specificity     F1            Balanced.Accuracy
# Random Forest with oversampling   0.5510204 0.9795573         0.7584126
# Random Forest with weights        0.4489796 0.9810207         0.7094773
# Decision Tree                     0.5918367 0.9100497         0.7157126
# Linear Model                      0.7551020 0.6551146         0.6219213
# 
# The final model evaluated against the validation set shows fairly similar results.
#
# Specificity                F1 Balanced Accuracy 
# 0.4918033         0.9826931         0.7330668 
# 
#----------------------------------->




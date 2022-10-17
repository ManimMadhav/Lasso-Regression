# load required packages
library(data.table) # used for reading and manipulation of data
library(dplyr)	    # used for data manipulation and joining
library(glmnet)	    # used for regression
library(ggplot2)    # used for ploting
library(caret)	    # used for modeling
library(xgboost)    # used for building XGBoost model
library(e1071)	    # used for skewness
library(cowplot)    # used for combining multiple plots

# Loading data
train = fread("Train.csv")
test = fread("Test.csv")

# Setting test dataset
# Combining datasets
# add Item_Outlet_Sales to test data
test[, Item_Outlet_Sales := NA]

combi = rbind(train, test)

# Fill missing values
missing_index = which(is.na(combi$Item_Weight))
for(i in missing_index)
{
  item = combi$Item_Identifier[i]
  combi$Item_Weight[i] =
    mean(combi$Item_Weight[combi$Item_Identifier == item],
         na.rm = T)
}

# Replacing 0 in Item_Visibility with mean
zero_index = which(combi$Item_Visibility == 0)
for(i in zero_index)
{
  item = combi$Item_Identifier[i]
  combi$Item_Visibility[i] =
    mean(combi$Item_Visibility[combi$Item_Identifier == item],
         na.rm = T)
}

# Label Encoding
# To convert categorical in numerical
combi[, Outlet_Size_num := ifelse(Outlet_Size == "Small", 0,
                                  ifelse(Outlet_Size == "Medium",
                                         1, 2))]

combi[, Outlet_Location_Type_num :=
        ifelse(Outlet_Location_Type == "Tier 3", 0,
               ifelse(Outlet_Location_Type == "Tier 2", 1, 2))]

combi[, c("Outlet_Size", "Outlet_Location_Type") := NULL]

# One Hot Encoding
# To convert categorical in numerical
ohe_1 = dummyVars("~.", data = combi[, -c("Item_Identifier",
                                          "Outlet_Establishment_Year",
                                          "Item_Type")], fullRank = T)
ohe_df = data.table(predict(ohe_1, combi[, -c("Item_Identifier",
                                              "Outlet_Establishment_Year",
                                              "Item_Type")]))

combi = cbind(combi[, "Item_Identifier"], ohe_df)

# Remove skewness
skewness(combi$Item_Visibility)
skewness(combi$price_per_unit_wt)

# log + 1 to avoid division by zero
combi[, Item_Visibility := log(Item_Visibility + 1)]

# Scaling and Centering data
num_vars = which(sapply(combi, is.numeric)) # index of numeric features
num_vars_names = names(num_vars)

combi_numeric = combi[, setdiff(num_vars_names,
                                "Item_Outlet_Sales"),
                      with = F]

prep_num = preProcess(combi_numeric,
                      method=c("center", "scale"))
combi_numeric_norm = predict(prep_num, combi_numeric)

# removing numeric independent variables
combi[, setdiff(num_vars_names,
                "Item_Outlet_Sales") := NULL]
combi = cbind(combi, combi_numeric_norm)

# splitting data back to train and test
train = combi[1:nrow(train)]
test = combi[(nrow(train) + 1):nrow(combi)]

# Removing Item_Outlet_Sales
test[, Item_Outlet_Sales := NULL]

# Model Building :Lasso Regression
set.seed(123)
control = trainControl(method ="cv", number = 5)
Grid_la_reg = expand.grid(alpha = 1,
                          lambda = seq(0.001, 0.1, by = 0.0002))

# Training lasso regression model
lasso_model = train(x = train[, -c("Item_Identifier",
                                   "Item_Outlet_Sales")],
                    y = train$Item_Outlet_Sales,
                    method = "glmnet",
                    trControl = control,
                    tuneGrid = Grid_la_reg
)

print(lasso_model)

# mean validation score
print(mean(lasso_model$resample$RMSE))

# Plot
plot(lasso_model, main = "Lasso Regression")

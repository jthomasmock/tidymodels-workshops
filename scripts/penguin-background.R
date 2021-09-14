## MODEL WITH VFOLD_CV
## Run this as a background job in RStudio

library(palmerpenguins)
library(tidymodels)
library(tidyverse)

penguins <- na.omit(penguins)
split_data <- initial_split(penguins, 0.75)

train_data <- training(split_data)
test_data <- testing(split_data)

my_rec <- recipe(
  species ~ sex + bill_length_mm + flipper_length_mm + body_mass_g,
  data = train_data
)

rf_mod <- rand_forest(trees = 100) %>%
  set_engine(
    "ranger",
    importance = "impurity", # variable importance
    num.threads = 4          # Parallelize
  ) %>%
  set_mode("classification")

rf_wflow <- workflow() %>%
  add_recipe(my_rec) %>% # Same recipe
  add_model(rf_mod) # New model

set.seed(37)
pen_folds <- vfold_cv(train_data, v = 10, repeats = 5)

# verbose - this will print output when in a background job
keep_pred <- control_resamples(save_pred = TRUE, verbose = TRUE)

set.seed(37)
rf_res <- fit_resamples(rf_wflow, resamples = pen_folds, control = keep_pred)

# save output but also returns to global environment
rf_res %>%
  write_rds("rf-resampled.rds")

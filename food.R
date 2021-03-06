library(dplyr)
library(readr)
# library(purrr)
# library(tidyverse)
library(keras)

reticulate::use_condaenv("labels")

## data & generator prep ----
labels_df <- read_csv("food/train/train.csv") %>% 
  mutate(Expected = as.character(Expected))
  # mutate(Expected = list(Expected))
# labels_cat <- to_categorical(labels_df$Expected)
# df_to_flow_from <- tibble(
#   x = labels_df$Id,
#   y = as.array(labels_cat)
# )
img_path <- "food/train/images"

img_gen <- image_data_generator(
  # preprocessing_function = scale,
  # featurewise_center = TRUE,
  samplewise_center = TRUE,
  # featurewise_std_normalization = TRUE,
  samplewise_std_normalization = TRUE,
  rotation_range = 120,
  vertical_flip = TRUE,
  horizontal_flip = TRUE,
  rescale = 1/255, 
  validation_split = .2
) 

training_generator <- flow_images_from_dataframe(
  ## sparse-ish:
  dataframe = labels_df,
  directory = img_path,
  x_col = "Id",
  y_col = "Expected",
  
  ## one-hot(ish):
  # dataframe = df_to_flow_from,
  # directory = img_path,
  # x_col = "x",
  # y_col = "y",
  
  generator = img_gen, 
  subset = "training",
  target_size = c(224, 224), color_mode = "rgb",
  # classes = NULL, class_mode = "categorical", 
  batch_size = 8
) 

# have a look at one batch:
nx <- training_generator %>% generator_next()

validation_generator <- flow_images_from_dataframe(
  dataframe = labels_df, 
  directory = img_path,
  x_col = "Id", 
  y_col = "Expected",
  generator = img_gen, 
  subset = "validation",
  target_size = c(224, 224), color_mode = "rgb",
  # classes = NULL, class_mode = "sparse", 
  batch_size = 2
)

## xception model prep ----
xcpt <- application_xception(
  include_top = FALSE,
  weights = "imagenet",
  input_shape = c(224, 224, 3)
) %>% 
  # set layers to untrainable
  freeze_weights()

preds <- xcpt$output %>%  
  layer_flatten() %>% 
  # layer_global_average_pooling_2d() %>% 
  layer_dense(1024, activation = "relu") %>% 
  layer_dense(512, activation = "relu") %>% 
  # layer_dense(256, activation = "relu") %>% 
  layer_dense(107, activation = "softmax") 

model <- keras_model(
  inputs = xcpt$input,
  outputs = preds
)

model %>% 
  compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_sgd(lr = .001, momentum = .8),
    metrics = "accuracy"
  )

early_stopping <- callback_early_stopping(monitor = 'val_loss', patience = 2)


## training ----
done <- model %>% 
  fit_generator(
    training_generator,
    steps_per_epoch = 107,
    epochs = 4,
    callbacks = early_stopping,
    validation_data = validation_generator,
    validation_steps = 107,
    verbose = 1
  )

plot(done)

## prdictions for test data ----
test_path <- "food/test/images"
test_df   <- tibble(
  Id = list.files(test_path)
)

prediction_generator <- flow_images_from_dataframe(
  dataframe = test_df, 
  directory = test_path,
  x_col = "Id", 
  y_col = "Expected",
  generator = image_data_generator(rescale = 1/255), 
  target_size = c(224, 224), color_mode = "rgb",
  classes = NULL, class_mode = NULL,
  batch_size = 2
)

predded <- model %>% 
  predict_generator(
    prediction_generator,
    steps = 4007
  )

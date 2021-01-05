library(dplyr)
library(readr)
# library(tidyverse)
library(keras)

reticulate::use_condaenv("labels")

## data & generator prep ----
labels_df <- read_csv("food/train/train.csv") %>% mutate(Expected = as.character(Expected))
labels_cat <- to_categorical(labels_df$Expected)
df_to_flow_from <- tibble(
  x = labels_df$Id,
  y = labels_cat
)
img_path <- "food/train/images"

img_gen <- image_data_generator(
  featurewise_center = TRUE,
  samplewise_center = TRUE,
  featurewise_std_normalization = TRUE,
  samplewise_std_normalization = TRUE,
  #rotation_range = sample(c(0, 90, 180)),
  #vertical_flip = sample(c(TRUE, FALSE)),
  #horizontal_flip = sample(c(TRUE, FALSE)),
  rescale = 1/255, 
  validation_split = .2
) 

training_generator <- flow_images_from_dataframe(
  dataframe = labels_df, 
  directory = img_path,
  x_col = "Id", 
  y_col = "Expected",
  generator = img_gen, 
  subset = "training",
  target_size = c(224, 224), color_mode = "rgb",
  # classes = NULL, class_mode = "categorical", 
  batch_size = 8
) 

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

## model prep ----
# vgg19 <- application_vgg19(
#   include_top = FALSE,
#   weights = "imagenet",
#   input_shape = c(224, 224, 3)
# )

rn50 <- application_resnet50(
  include_top = FALSE,
  weights = "imagenet",
  input_shape = c(224, 224, 3)
)

# set layers to untrainable
for (layer in rn50$layers) {
  layer$trainable <- FALSE
}

preds <- rn50$output %>%  
  layer_flatten() %>% 
  layer_dense(512, activation = "relu") %>% 
  layer_dense(256, activation = "relu") %>% 
  layer_dense(107, activation = "softmax") 

model <- keras_model(
  inputs = rn50$input,
  outputs = preds
)

model %>% 
  compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = optimizer_sgd(lr = .001, momentum = .9),
    metrics = "accuracy"
  )


## training ----
done <- model %>% 
  fit_generator(
    training_generator,
    steps_per_epoch = 107,
    epochs = 2,
    validation_data = validation_generator,
    validation_steps = 107,
    verbose = 1
  )

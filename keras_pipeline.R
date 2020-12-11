library(tidyverse)
library(keras)

theme_set(hrbrthemes::theme_modern_rc())
reticulate::use_condaenv("labels")

### data & label preparation ----
# data
pic_paths  <- paste0("pics/", list.files("pics/"))
imgs       <- map(pic_paths, image_load)
img_arrays <- map(imgs, image_to_array)

arrays_reshaped <- array_reshape(img_arrays, c(length(pic_paths), 263, 263, 3))
train_data      <- imagenet_preprocess_input(arrays_reshaped)

# labels
labs <- rep(c(0, 1), each = length(pic_paths)/2)
train_labels <- to_categorical(labs, num_classes = 2)

# some housekeeping
rm(arrays_reshaped, img_arrays, imgs, labs)


### model preparation ----
vgg16 <- application_vgg16(
  include_top = FALSE,
  weights = "imagenet",
  # input_tensor = NULL,
  input_shape = c(263, 263, 3)
  # pooling = NULL,
  # classes = 1000
)

preds <- vgg16$output %>%
  layer_flatten() %>% 
  # layer_dense(256, activation = "relu", input_shape = 263*263*3) %>% 
  layer_dense(512, activation = "relu") %>% 
  layer_dense(128, activation = "relu") %>% 
  layer_dense(2, activation = "sigmoid")

model <- keras_model(inputs = vgg16$input,
                     outputs = preds)

model %>% 
# m %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_adam(lr = .001, decay = .001),
    metrics = "accuracy"
  )

rm(vgg16, preds)
### fit vgg16-model ----
model %>% 
# m %>% 
  fit(
    train_data,
    train_labels,
    epochs = 5,
    # batch_size = 10,
    validation_split = .2,
    verbose = 2,
    shuffle = TRUE
  )

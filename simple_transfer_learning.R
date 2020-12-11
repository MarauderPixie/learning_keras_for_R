library(EBImage)
library(tidyverse)
library(keras)

theme_set(hrbrthemes::theme_modern_rc())

### some crude data prep ----
pic_paths <- paste0("pics/", list.files("pics/"))

pics <- map(pic_paths, readImage)
labs <- rep(c(0, 1), each = length(pic_paths)/2)

pics_reshaped <- map(seq_along(pics), function(x){
  array_reshape(pics[[x]], c(263, 263, 3))
})


pics_ready <- NULL
for (i in seq_along(pics)){
  pics_ready <- rbind(pics_ready, pics_reshaped[[i]])
}

# num_classes is optional, it seems
train_labels <- to_categorical(labs, num_classes = 2)

test <- array(pics_ready, c(length(pics), 263, 263, 3))

rm(pics, pics_ready, pics_reshaped, i, labs, pic_paths)

### to the modeling ----

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
  # layer_dense(512, activation = "relu") %>% 
  layer_dense(128, activation = "relu") %>% 
  layer_dense(2, activation = "softmax")


model <- keras_model(inputs = vgg16$input,
                     outputs = preds)


# basically the same as before
model %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_adam(lr = .05, decay = .005),
    metrics = "accuracy"
  )

model %>% 
  fit(
    test2,
    train_labs,
    epochs = 30,
    # batch_size = 19,
    validation_split = .2,
    verbose = 2,
    shuffle = TRUE
  )

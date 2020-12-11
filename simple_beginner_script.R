library(EBImage)
library(tidyverse)
library(keras)

theme_set(hrbrthemes::theme_modern_rc())

pic_paths <- paste0("pics/", list.files("pics/"))

pics <- map(pic_paths, readImage)
labs <- rep(c(0, 1), each = length(pics)/2)

pics_reshaped <- map(seq_along(pics), function(x){
  array_reshape(pics[[x]], c(263, 263, 3))
})


pics_ready <- NULL
for (i in seq_along(pics)){
  pics_ready <- rbind(pics_ready, pics_reshaped[[i]])
}

# num_classes is optional, it seems
train_labs <- to_categorical(labs, num_classes = 2)

test <- array(pics_ready, c(length(pics), 263, 263, 3))

model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 64, kernel_size = c(12, 12), 
                activation = "relu", 
                input_shape = c(263, 263, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(4, 4)) %>% 
  
  layer_conv_2d(filters = 128, kernel_size = c(12, 12), 
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  
  layer_conv_2d(filters = 128, kernel_size = c(12, 12),
                activation = "relu") %>%
  
  layer_flatten() %>% 
  # layer_dense(256, activation = "relu", input_shape = 263*263*4) %>% 
  layer_dense(64, activation = "relu") %>% 
  layer_dense(2, activation = "softmax")

summary(model)

model %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_adam(lr = .00001, decay = .01),
    metrics = "accuracy"
  )

mtraind <- model %>% 
  fit(test,
      train_labs,
      epochs = 50,
      batch_size = 3,
      validation_split = .2,
      verbose = 2,
      shuffle = TRUE)

model %>% 
  predict_classes(test)

model %>% 
  evaluate(test,
           train_labs)


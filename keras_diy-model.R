m <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 64, kernel_size = c(5, 5), 
                activation = "relu", 
                input_shape = c(263, 263, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  
  layer_conv_2d(filters = 128, kernel_size = c(5, 5), 
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  
  layer_conv_2d(filters = 128, kernel_size = c(5, 5), 
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  
  layer_conv_2d(filters = 128, kernel_size = c(5, 5), 
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  
  # layer_conv_2d(filters = 128, kernel_size = c(12, 12),
  #               activation = "relu") %>%
  
  layer_flatten() %>% 
  # layer_dense(256, activation = "relu", input_shape = 263*263*4) %>% 
  layer_dense(512, activation = "relu") %>% 
  layer_dense(2, activation = "sigmoid")

m %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_adam(), # lr = .05, decay = .005),
    metrics = "accuracy"
  )

m %>% 
  fit(
    train_data,
    train_labels,
    epochs = 30,
    # batch_size = 10,
    validation_split = .2,
    verbose = 2,
    shuffle = TRUE
  )

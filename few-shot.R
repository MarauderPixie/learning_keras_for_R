library(tidyverse)
library(keras)

theme_set(hrbrthemes::theme_modern_rc())
# reticulate::use_condaenv("labels")

lab_path <- paste0("few-shot/labels/", list.files("few-shot/labels/"))
img_path <- paste0("few-shot/pics/", list.files("few-shot/pics/"))

data_labels <- map_df(lab_path, function(x) {
  suppressMessages(
    read_table2(x, col_names = FALSE)
  )
}) %>% 
  transmute(
    x1 = X1 / 960,
    y1 = X2 / 540,
    x2 = X3 / 960,
    y2 = X4 / 540
    # area = (x2 - x1) * (y2 - y1)
  )

img_files  <- map(img_path, image_load)
img_arrays <- map(img_files, image_to_array)

arrays_reshaped <- array_reshape(img_arrays, c(length(img_path), 540, 960, 3)) / 255
train_data      <- imagenet_preprocess_input(arrays_reshaped, mode = "tf")

########
array_labels <- array(c(data_labels$x1, data_labels$y1, data_labels$x2, data_labels$y2), c(10, 4))
########

train <- list(
  data = arrays_reshaped,
  # data = train_data,
  # labs = data_labels
  labs = array_labels
)

rm(img_arrays, img_files, arrays_reshaped, train_data, 
   data_labels, array_labels, lab_path, img_path)


## model, I guess?
vgg16 <- application_vgg16(
  include_top = FALSE,
  weights = "imagenet",
  # input_tensor = NULL,
  input_shape = c(540, 960, 3)
  # pooling = NULL,
  # classes = 1000
)

rn50 <- application_resnet50(
  include_top = FALSE, 
  input_shape = c(540, 960, 3)
)

# set layers to untrainable
for (layer in vgg16$layers) {
  layer$trainable <- FALSE
}

preds <- vgg16$output %>%
# preds <- rn50$output %>%  
  layer_flatten() %>% 
  # layer_dense(256, activation = "relu", input_shape = 263*263*3) %>% 
  # layer_dense(512, activation = "relu") %>% 
  layer_dense(128, activation = "relu") %>% 
  layer_dense(64, activation = "relu") %>% 
  layer_dense(32, activation = "relu") %>% 
  layer_dense(4, activation = "sigmoid")

model <- keras_model(
  inputs = vgg16$input,
  # inputs = rn50$input,
  outputs = preds
)

model %>% 
  compile(
    loss = "mean_squared_error",
    optimizer = optimizer_sgd() # lr = .001, decay = .001),
    # metrics = "accuracy"
  )

## aaaaand train
done <- model %>% 
  fit(
    train$data,
    train$labs,
    epochs = 10,
    batch_size = 4,
    validation_split = .2,
    verbose = 2,
    shuffle = TRUE
  )

# and why not; save the model, for whatever it's worth
save_model_tf(model, "few-shot/models/")

# load via:
# model <- load_model_tf("few-shot/models/")



#######
## and now to data generators-ing
# train_generator <- flow_images_from_directory("few-shot/", generator = image_data_generator(),
#                                               target_size = c(940, 540), color_mode = "rgb",
#                                               class_mode = "sparse", batch_size = 4, shuffle = TRUE)
# 
# validation_generator <- flow_images_from_directory("few-shot/pics/", generator = image_data_generator(),
#                                                    target_size = c(940, 540), color_mode = "rgb", classes = NULL,
#                                                    class_mode = "sparse", batch_size = 2, shuffle = TRUE)


p_gen <- flow_images_from_directory("few-shot/", 
                                    image_data_generator(rescale = 1/255), #, preprocessing_function = preproc),
                                    target_size = c(960, 540), color_mode = "rgb",
                                    classes = NULL, class_mode = NULL, batch_size = 5)

model %>% 
  predict_generator(p_gen, steps = 2)

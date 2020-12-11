# preproc <- function(x){
#   img_file  <- image_load(x)
#   img_array <- image_to_array(img_file)
#   array_rs  <- array_reshape(img_array, c(1, 540, 960, 3)) / 255
#   return(array_rs)
# }


## wrong approach, maybe?
f_gen <- flow_images_from_directory("few-shot/", # preproc(), 
                                    image_data_generator(rescale = 1/255),
                                    target_size = c(960, 540), color_mode = "rgb",
                                    classes = NULL, class_mode = "sparse", batch_size = 5)

fnx <- generator_next(f_gen)


## flow from dataframe
df_to_flow_from <- tibble(
  x_col = list.files("few-shot/pics/"),
  data_labels
)

img_gen <- image_data_generator(rescale = 1/255, validation_split = .2)

f_gen <- flow_images_from_dataframe(
  df_to_flow_from, "few-shot/pics/",
  "x_col", c("x1", "y1", "x2", "y2"),
  img_gen, subset = "training",
  target_size = c(540, 960), color_mode = "rgb",
  classes = NULL, class_mode = "other", batch_size = 4
)

v_gen <- flow_images_from_dataframe(
  df_to_flow_from, "few-shot/pics/",
  "x_col", c("x1", "y1", "x2", "y2"),
  img_gen, subset = "validation",
  target_size = c(540, 960), color_mode = "rgb",
  classes = NULL, class_mode = "other", batch_size = 2
)

## train on this?
model %>% 
  fit_generator(f_gen,
                steps_per_epoch = 2,
                epochs = 5,
                validation_data = v_gen,
                validation_steps = 1,
                verbose = 2)

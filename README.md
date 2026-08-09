# README


## Neural Network Architecture
     Input Layer                  Hidden Layer                  Output Layer
    (28x28 pixels)              (e.g., 128 ReLU)              (10 Digit Classes)
      
      [ x_1 ]  \                  / [ h_1 ] \                  / [ y_0 ] (Digit 0)
      [ x_2 ]   \                /  [ h_2 ]  \                /  [ y_1 ] (Digit 1)
      [  .  ]    \   W_1        /   [  .  ]   \   W_2        /   [  .  ]    .
      [  .  ]     * * * * * * *     [  .  ]    * * * * * * *     [  .  ]    .
      [  .  ]    / (784 x H)    \   [  .  ]   /   (H x 10)  \    [  .  ]    .
      [ x_784 ] /                \  [ h_H ]  /               \   [ y_9 ] (Digit 9)
      
     784 neurons                   H neurons                    10 neurons
  (Flattened Image)             (Latent Features)            (Softmax Output)
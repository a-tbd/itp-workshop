```
usage: neural_style.py [-h] --content CONTENT --styles STYLE [STYLE ...]
                           --output OUTPUT [--iterations ITERATIONS]
                           [--print-iterations PRINT_ITERATIONS]
                           [--checkpoint-output OUTPUT]
                           [--checkpoint-iterations CHECKPOINT_ITERATIONS]
                           [--width WIDTH]
                           [--style-scales STYLE_SCALE [STYLE_SCALE ...]]
                           [--network VGG_PATH]
                           [--content-weight-blend CONTENT_WEIGHT_BLEND]
                           [--content-weight CONTENT_WEIGHT]
                           [--style-weight STYLE_WEIGHT]
                           [--style-layer-weight-exp STYLE_LAYER_WEIGHT_EXP]
                           [--style-blend-weights STYLE_BLEND_WEIGHT [STYLE_BLEND_WEIGHT ...]]
                           [--tv-weight TV_WEIGHT] [--learning-rate LEARNING_RATE]
                           [--beta1 BETA1] [--beta2 BETA2] [--eps EPSILON]
                           [--initial INITIAL]
                           [--initial-noiseblend INITIAL_NOISEBLEND]
                           [--preserve-colors] [--pooling POOLING]
    
    optional arguments:
      -h, --help            show this help message and exit
      --content CONTENT     content image
      --styles STYLE [STYLE ...]
                            one or more style images
      --output OUTPUT       output path
      --iterations ITERATIONS
                            iterations (default 1000)
      --print-iterations PRINT_ITERATIONS
                            statistics printing frequency
      --checkpoint-output OUTPUT
                            checkpoint output format, e.g. output%s.jpg
      --checkpoint-iterations CHECKPOINT_ITERATIONS
                            checkpoint frequency
      --width WIDTH         output width
      --style-scales STYLE_SCALE [STYLE_SCALE ...]
                            one or more style scales
      --network VGG_PATH    path to network parameters (default imagenet-vgg-
                            verydeep-19.mat)
      --content-weight-blend CONTENT_WEIGHT_BLEND
                            content weight blend, conv4_2 * blend + conv5_2 *
                            (1-blend) (default 1)
      --content-weight CONTENT_WEIGHT
                            content weight (default 5.0)
      --style-weight STYLE_WEIGHT
                            style weight (default 500.0)
      --style-layer-weight-exp STYLE_LAYER_WEIGHT_EXP
                            style layer weight exponentional increase -
                            weight(layer<n+1>) = weight_exp*weight(layer<n>)
                            (default 1)
      --style-blend-weights STYLE_BLEND_WEIGHT [STYLE_BLEND_WEIGHT ...]
                            style blending weights
      --tv-weight TV_WEIGHT
                            total variation regularization weight (default 100.0)
      --learning-rate LEARNING_RATE
                            learning rate (default 10.0)
      --beta1 BETA1         Adam: beta1 parameter (default 0.9)
      --beta2 BETA2         Adam: beta2 parameter (default 0.999)
      --eps EPSILON         Adam: epsilon parameter (default 1e-08)
      --initial INITIAL     initial image
      --initial-noiseblend INITIAL_NOISEBLEND
                            ratio of blending initial image with normalized noise
                            (if no initial image specified, content image is used)
                            (default None)
      --preserve-colors     style-only transfer (preserving colors) - if color
                            transfer is not needed
      --pooling POOLING     pooling layer configuration: max or avg (default max)
      ```
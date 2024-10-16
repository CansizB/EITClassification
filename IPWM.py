import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import Model
from classification_models_3D.tfkeras import Classifiers


def Net(model_name: str):
    
    """
    model_name: Defines the model that should be selected based on the intended usage.

    The model can be 'densenet201', 'resnet50', etc. 
    to getting more information please refer to "https://github.com/ZFTurbo/classification_models_3D" 
    
    """

    net, preprocess_input = Classifiers.get(model_name)
    
    model = net(input_shape=(330, 32, 32, 3), include_top=False, weights="imagenet")

    for layer in model.layers:
        layer.trainable = False
    out1 = model.output
    
    out1=layers.GlobalAveragePooling3D()(out1)
    
    Dense_model = Model(inputs=[model.input], outputs=[out1])
    Dense_model.summary()

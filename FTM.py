import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import Model


def Net(model_name: str):
    
    """
    model_name: Defines the model that should be selected based on the intended usage.

    The model can be 'densenet201', 'resnet50', etc. 
    to getting more information please refer to "https://github.com/ZFTurbo/classification_models_3D" 
    
    """

    net, preprocess_input = Classifiers.get(model_name)
    
    model = net(input_shape=(330, 32, 32, 3), include_top=False, weights="imagenet")
    
    c=0 
    for layer in model.layers:
        c=c+1
    
    Nb_Layers=c
    OUT_LAYERS = Nb_Layers * (10/100)
    OUT_LAYERS = round(OUT_LAYERS,0)  
    
    for layer in model.layers[:-int(OUT_LAYERS)]:
        layer.trainable = False
    
    out1 = model.output
    
    out1=layers.GlobalAveragePooling3D()(out1)
    
    out1 = layers.Dense(2048, activation = "relu")(out1)
    predictions = layers.Dense(5, activation = "softmax")(out1)
    
    
    FTM = Model(inputs=[model.input], outputs=[predictions])
    
    FTM.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy",tf.keras.metrics.F1Score(average="macro")])
    
    return FTM

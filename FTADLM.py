from classification_models_3D.tfkeras import Classifiers


def Net(model_name: str):

    """
    
    model_name: Defines the model that should be selected based on the intended usage.
    The model name can be 'densenet201', 'resnet50', etc. 
    
    For more information, please refer to:
    https://github.com/ZFTurbo/classification_models_3D

    To install the correct version, run:
    pip install classification-models-3D==1.0.10
    
    
    """

    net, preprocess_input = Classifiers.get(model_name)
    
    model = net(input_shape=(330, 32, 32, 3), include_top=False, weights="imagenet")
    
    c=0 
    for layer in model.layers:
        c=c+1
    
    Nb_Layers=c
    OUT_LAYERS = Nb_Layers * (15/100)
    OUT_LAYERS = round(OUT_LAYERS,0)  
    
    b=0
    for layer in model.layers[:-int(OUT_LAYERS)]:
        b=b+1
        layer.trainable = False
    
    out1 = model.output
    
    out1=layers.GlobalAveragePooling3D()(out1)
    
    out1 = layers.Dense(2048, activation = "relu")(out1)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation = "relu")(x)
    predictions = layers.Dense(5, activation = "softmax")(out1)
    
    
    FTADLM = Model(inputs=[model.input], outputs=[predictions])
    
    FTADLM.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy",tf.keras.metrics.F1Score(average="macro")])
    
    
    
    return FTADLM

from classification_models_3D.tfkeras import Classifiers


def Net(model_name: str):

    '''
    model_name: it defines the model, that should be selected depend on the usage.  it can be 'densenet201', 'resnet50' etc. 
    '''

    net, preprocess_input = Classifiers.get(model_name)
    
    model = densenet201(input_shape=(330, 32, 32, 3), include_top=False, weights="imagenet")
    
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

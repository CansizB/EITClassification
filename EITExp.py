import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from classification_models_3D.tfkeras import Classifiers

class ModelTypes:
    def __init__(self, input_shape=(330, 32, 32, 3)):
        self.input_shape = input_shape

    def build_net(self, model_name: str):
        """
        model_name: Defines the model that should be selected based on the intended usage.

        The model can be 'densenet201', 'resnet50', etc.
        """
        net, preprocess_input = Classifiers.get(model_name)
        model = net(input_shape=self.input_shape, include_top=False, weights="imagenet")

        return model 

    def IPWM(self, model_name: str):
        """
        Builds and compiles IPWM.
        """

        for layer in model.layers:
            layer.trainable = False

        out = layers.GlobalAveragePooling3D()(model.output)
        model = self.build_net(model_name)
      
        model = Model(inputs=[model.input], outputs=[out])
      
        model.compile(optimizer="adam", loss='categorical_crossentropy',
                      metrics=["accuracy", tf.keras.metrics.F1Score(average="macro")])
        return model

    def FTM(self, model_name: str, nb_classes: int):
        """
        Builds and compiles the FTM.
        """
        model = self.build_net(model_name)
        
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
        predictions = layers.Dense(nb_classes, activation = "softmax")(out1)
        
        
        model = Model(inputs=[model.input], outputs=[predictions])
        
        model.compile(optimizer="adam", loss='categorical_crossentropy',
                      metrics=["accuracy", tf.keras.metrics.F1Score(average="macro")])
        return model

    def FTADLM(self, model_name:str, nb_classes: int):
        """
        Builds and compiles the FTADLM.
        """
        model = self.build_net(model_name)
      
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
        predictions = layers.Dense(nb_classes, activation = "softmax")(out1)
        
        
        model = Model(inputs=[model.input], outputs=[predictions])
        
        model.compile(optimizer="adam", loss='categorical_crossentropy',
                      metrics=["accuracy", tf.keras.metrics.F1Score(average="macro")])
        return model


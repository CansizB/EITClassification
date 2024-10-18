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
        The model name can be 'densenet201', 'resnet50', etc. 
        
        For more information, please refer to:
        https://github.com/ZFTurbo/classification_models_3D
    
        To install the correct version, run:
        pip install classification-models-3D==1.0.10
        """
        net, preprocess_input = Classifiers.get(model_name)
        model = net(input_shape=self.input_shape, include_top=False, weights="imagenet")

        return model 

    def IPWM(self, model_name: str):
        """
        Builds and compiles IPWM.
        """

        # Build the model first
        model = self.build_net(model_name)

        # Freeze all layers
        for layer in model.layers:
            layer.trainable = False

        # Add global pooling layer and build the model
        out = layers.GlobalAveragePooling3D()(model.output)
        model = Model(inputs=[model.input], outputs=[out])
      
        model.compile(optimizer="adam", loss='categorical_crossentropy',
                      metrics=["accuracy", tf.keras.metrics.F1Score(average="macro")])
        return model

    def FTM(self, model_name: str, nb_classes: int):
        """
        Builds and compiles the FTM.
        """
        model = self.build_net(model_name)
        
        # Count the number of layers and define the number of layers to freeze
        Nb_Layers = len(model.layers)
        OUT_LAYERS = round(Nb_Layers * 0.1)  # 10% of the layers
        
        # Freeze first 90% of the layers
        for layer in model.layers[:-OUT_LAYERS]:
            layer.trainable = False
        
        # Build the head of the network
        out = layers.GlobalAveragePooling3D()(model.output)
        out = layers.Dense(2048, activation="relu")(out)
        predictions = layers.Dense(nb_classes, activation="softmax")(out)
        
        # Create and compile the model
        model = Model(inputs=[model.input], outputs=[predictions])
        
        model.compile(optimizer="adam", loss='categorical_crossentropy',
                      metrics=["accuracy", tf.keras.metrics.F1Score(average="macro")])
        return model

    def FTADLM(self, model_name: str, nb_classes: int):
        """
        Builds and compiles the FTADLM.
        """
        model = self.build_net(model_name)
      
        # Count the number of layers and define the number of layers to freeze
        Nb_Layers = len(model.layers)
        OUT_LAYERS = round(Nb_Layers * 0.15)  # 15% of the layers
        
        # Freeze first 85% of the layers
        for layer in model.layers[:-OUT_LAYERS]:
            layer.trainable = False
        
        # Build the head of the network
        out = layers.GlobalAveragePooling3D()(model.output)
        out = layers.Dense(2048, activation="relu")(out)
        out = layers.Dropout(0.2)(out)
        out = layers.Dense(64, activation="relu")(out)
        predictions = layers.Dense(nb_classes, activation="softmax")(out)
        
        # Create and compile the model
        model = Model(inputs=[model.input], outputs=[predictions])
        
        model.compile(optimizer="adam", loss='categorical_crossentropy',
                      metrics=["accuracy", tf.keras.metrics.F1Score(average="macro")])
        return model

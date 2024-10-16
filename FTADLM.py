from classification_models_3D.tfkeras import Classifiers

densenet201, preprocess_input = Classifiers.get('densenet201')

model = densenet201(input_shape=(330, 32, 32, 3), include_top=False, weights="imagenet")

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


model1 = Model(inputs=[model.input], outputs=[predictions])

model1.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy",tf.keras.metrics.F1Score(average="macro")])



model1.summary()


if __name__ == "__main__":

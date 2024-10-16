def ftExp(ExpName: str, nb_epoch: int, nb_classes: int):
  
  for i in range(5):
  
    c= str(i+1)
    
    y_train= np.load("path/fold_" + c + "/y_train.npy")
    y_test= np.load("path/fold_" + c + "/y_test.npy")
    
    X_train= np.load("path/fold_" + c + "/X_train.npy")
    X_test= np.load("path/fold_" + c + "/X_test.npy")
    
    X_train3d= makeRGB(X_train, 330)
    X_test3d= makeRGB(X_test,330)
    
    X_train3d, y_train = shuffle(X_train3d, np.array(y_train), random_state=42)
    
    y_trainC = tf.keras.utils.to_categorical(y_train, num_classes=nb_classes)
    y_testC = tf.keras.utils.to_categorical(y_test, num_classes=nb_classes)
    
    history = net.fit(X_train3d, y_trainC, epochs=nb_epoch, batch_size=64, validation_data=(X_test3d,y_testC))
    
    net.save(ExpName + "Fold" + c + ".keras")
    
    del net


def modelImpFT1(foldId):

    Id = str(foldId)

    Ftmodel = tf.keras.models.load_model('/Users/berkecansiz/Desktop/BRACETS/EIT/Classification_Models_3D_deneme1/Densenet201/5C/FT_Models/FT_Dense201Fold'+Id+ ".keras")

    pretrained_layers = Ftmodel.layers[:-1] 

    Ftmodel = Model(inputs=Ftmodel.input, outputs=pretrained_layers[-1].output)

    Ftmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return Ftmodel

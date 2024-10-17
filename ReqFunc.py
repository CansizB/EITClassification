class ImgRecon:
    def __init__(self, path):
        self.path = path
  
    def get_slice(self):
        a=pd.read_csv(self.path,header=None)
        i=np.array(a)
        i[np.isnan(i)] = 0
        i[i==0] = np.nan
        min_val = np.nanmin(i)
        max_val = np.nanmax(i)
        normalized_data = (i - min_val) / (max_val - min_val) * 255
    
    
        img_size = 32
        nb_img_vert = normalized_data.shape[0] // img_size
    
        imgs = []
    
        for j in range(nb_img_vert):
            
            img = normalized_data[j * img_size: (j + 1) * img_size]
            img[np.isnan(img)] = 0
            imgs.append(img)
    
        slice = np.stack(imgs, axis=0)
        return slice
    
    def RoiFrame(self, slice, start, end):
   
        frames=[]
        for i in range(start,end):
            frames.append(slice[i])
    
        frames=np.array(frames)
    
        return frames


    def KFOLD(self):

    def CSVtoNumPy(self, class_info: str, meta, file_name, CSVname)
      train = []
      test = []
      
      for i in range(0, 78):
          subject = i + 1
      
          if meta["Diagnosis"][i] == class_info:
      
              for j in range(len(file_name)):
      
                  if file_name["SubjectID"][j] == subject:
      
                      if file_name["SubjectID"][j] in train:
                          path = file_name["EITFilename"][j]
      
      
                          slice = get_slice(CSVname + "/" + path[:-4] + ".csv")
                          start = round(len(slice) / 2)-165
                          end = round(len(slice) / 2)+165
      
                          frames = RoiFrame(slice, start, end)
      
                          train.append(frames)
      
                      if file_name["SubjectID"][j] in test:
                          path = file_name["EITFilename"][j]
      
      
                          slice = get_slice(CSVname + "/" + path[:-4] + ".csv")
                          start = round(len(slice) / 2)-165
                          end = round(len(slice) / 2)+165
      
                          frames = RoiFrame(slice, start, end)
      
                          test.append(frames)
      
      train = np.array(train)
      test = np.array(test)
      return train, test
      
    def GenSet(self, path: str):
      train=train_idx_5
      test=test_idx_5

      CSV_name= path

      COPD_train, COPD_test = CSVtoNumPy("COPD")
      ILD_train, ILD_test = CSVtoNumPy("ILD")
      Healthy_train, Healthy_test = CSVtoNumPy("Healthy")
      Asthma_train, Asthma_test = CSVtoNumPy("Asthma")
      Infection_train, Infection_test = CSVtoNumPy("Infection")

      X_train=np.concatenate([COPD_train,ILD_train,Asthma_train,Infection_train,Healthy_train])
      X_test=np.concatenate([COPD_test,ILD_test,Asthma_test,Infection_test,Healthy_test])

      y_train= np.concatenate([np.ones(COPD_train.shape[0]),3*np.ones(ILD_train.shape[0]),np.zeros(Asthma_train.shape[0]),4*np.ones(Infection_train.shape[0]),2*np.ones(Healthy_train.shape[0])])
      y_test= np.concatenate([np.ones(COPD_test.shape[0]),3*np.ones(ILD_test.shape[0]),np.zeros(Asthma_test.shape[0]),4*np.ones(Infection_test.shape[0]),2*np.ones(Healthy_test.shape[0])])

      return X_train, X_test, y_train, y_test
      
      
      



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

    Ftmodel = tf.keras.models.load_model(ExpName + "Fold" +Id+ ".keras")

    pretrained_layers = Ftmodel.layers[:-1] 

    Ftmodel = Model(inputs=Ftmodel.input, outputs=pretrained_layers[-1].output)

    Ftmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return Ftmodel

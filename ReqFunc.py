import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.models import Model
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

class ImgRecon:

    def makeRGB(self, data, frame):
    
        data3d = np.zeros((data.shape[0],data.shape[1],data.shape[2],data.shape[3],3))
        
        for i in range(frame):
            for j in range(3):
                data3d[:, i, :, :, j] = data[:, i, :, :]
        data3d = data3d.astype(np.float32)
    
        return data3d
    


    def get_slice(self, path):
        a=pd.read_csv(path,header=None)
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

    def RoiFrame(self, slice_data, start, end):
        return np.array([slice_data[i] for i in range(start, end)])

    def KFOLD(self, nb_subj, meta_path, nb_classes=5):

        xph=np.random.random(nb_subj)
        meta=pd.read_csv(meta_path)
        meta.reset_index(drop=True, inplace=True)
        yph=meta["Diagnosis"]
        label_encoder = LabelEncoder()

        if nb_classes == 5:
            yph = label_encoder.fit_transform(yph)

        elif nb_classes == 3:
            yph = np.where(yph == 0, 0, np.where(yph == 1, 0, np.where(yph == 2, 1, 2))) #if 3 class structure is implemented 
        
        else:
            yph = np.where(yph == 2, 1, 0) #if binary structure is implemented 


        # Stratified K-Fold setup
        k = 5
        seed = 42
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        all_train_idx, all_test_idx = [], []

        for train_idx, test_idx in kf.split(xph, yph):
            all_train_idx.append(train_idx)
            all_test_idx.append(test_idx)

        # Create separate train/test indices for 5 folds
        self.train_test_splits = [(all_train_idx[i] + 1, all_test_idx[i] + 1) for i in range(5)]

    def CSVtoNumPy(self, class_info, mfile, CSVname, EITname, fold):
        train_idx, test_idx = self.train_test_splits[fold]
        
        train, test = [], []

        for i in range(0, 78):
            subject = i + 1
            if mfile["Diagnosis"][i] == class_info:
                for j in range(len(EITname)):
                    if EITname["SubjectID"][j] == subject:
                        path = EITname["EITFilename"][j]
                        slice_data = self.get_slice(CSVname + "/" + path[:-4] + ".csv")
                        start, end = round(len(slice_data) / 2) - 165, round(len(slice_data) / 2) + 165
                        frames = self.RoiFrame(slice_data, start, end)

                        if subject in train_idx:
                            train.append(frames)
                        if subject in test_idx:
                            test.append(frames)

        return np.array(train), np.array(test)

    def GenSet(self, fold, CSVname, mfile_path, EITname, nb_classes, save=False):
        CSVname = str(CSVname)
        mfile = pd.read_csv(str(mfile_path))
        EITname = pd.read_csv(str(EITname))

        self.KFOLD(nb_subj = 78, meta_path = mfile_path, nb_classes= nb_classes)
        # Generate datasets for different classes
        print ("COPD data are converting")
        COPD_train, COPD_test = self.CSVtoNumPy("COPD", mfile, CSVname, EITname, fold)
        print ("ILD data are converting")
        ILD_train, ILD_test = self.CSVtoNumPy("ILD", mfile, CSVname, EITname, fold)
        print ("Healthy data are converting")
        Healthy_train, Healthy_test = self.CSVtoNumPy("Healthy", mfile, CSVname, EITname, fold)
        print ("Asthma data are converting")
        Asthma_train, Asthma_test = self.CSVtoNumPy("Asthma", mfile, CSVname, EITname, fold)
        print ("Infection data are converting")
        Infection_train, Infection_test = self.CSVtoNumPy("Infection", mfile, CSVname, EITname, fold)

        print ("Data were converted from csv to numpy")
        # Concatenate and label the data
        X_train = np.concatenate([COPD_train, ILD_train, Asthma_train, Infection_train, Healthy_train])
        X_test = np.concatenate([COPD_test, ILD_test, Asthma_test, Infection_test, Healthy_test])
        
        if nb_classes == 5:
            y_train = np.concatenate([np.ones(COPD_train.shape[0]), 
                                  3 * np.ones(ILD_train.shape[0]), 
                                  np.zeros(Asthma_train.shape[0]), 
                                  4 * np.ones(Infection_train.shape[0]), 
                                  2 * np.ones(Healthy_train.shape[0])])
        
            y_test = np.concatenate([np.ones(COPD_test.shape[0]), 
                                     3 * np.ones(ILD_test.shape[0]), 
                                     np.zeros(Asthma_test.shape[0]), 
                                     4 * np.ones(Infection_test.shape[0]), 
                                     2 * np.ones(Healthy_test.shape[0])])
        elif nb_classes == 3:
            y_train = np.concatenate([np.zeros(COPD_train.shape[0]), 
                                    2 * np.ones(ILD_train.shape[0]), 
                                    np.zeros(Asthma_train.shape[0]), 
                                    2 * np.ones(Infection_train.shape[0]), 
                                    np.ones(Healthy_train.shape[0])])
        
            y_test = np.concatenate([np.zeros(COPD_test.shape[0]), 
                                     2 * np.ones(ILD_test.shape[0]), 
                                     np.zeros(Asthma_test.shape[0]), 
                                     2 * np.ones(Infection_test.shape[0]), 
                                     np.ones(Healthy_test.shape[0])])

        else:
            y_train = np.concatenate([np.zeros(COPD_train.shape[0]), 
                                    np.zeros(ILD_train.shape[0]), 
                                    np.zeros(Asthma_train.shape[0]), 
                                    np.zeros(Infection_train.shape[0]), 
                                    np.ones(Healthy_train.shape[0])])
        
            y_test = np.concatenate([np.zeros(COPD_test.shape[0]), 
                                     np.zeros(ILD_test.shape[0]), 
                                     np.zeros(Asthma_test.shape[0]), 
                                     np.zeros(Infection_test.shape[0]), 
                                     np.ones(Healthy_test.shape[0])])
    
        # Optionally save the arrays to .npy files
        if save:
            
            directory = f"data/C{nb_classes}/fold_{fold+1}"
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            np.save(os.path.join(directory, "X_train.npy"), X_train)
            np.save(os.path.join(directory, "X_test.npy"), X_test)
            np.save(os.path.join(directory, "y_train.npy"), y_train)
            np.save(os.path.join(directory, "y_test.npy"), y_test)

        print(f"fold {fold+1} data were preprocessed")
            
        return X_train, X_test, y_train, y_test



def ftExp(net, fold, ExpName, nb_epoch, nb_classes):
    #for i in range(5):
    fold = str(fold + 1)
    y_train = np.load(f"data/C{nb_classes}/fold_{fold}/y_train.npy")
    y_test = np.load(f"data/C{nb_classes}/fold_{fold}/y_test.npy")
    X_train = np.load(f"data/C{nb_classes}/fold_{fold}/X_train.npy")
    X_test = np.load(f"data/C{nb_classes}/fold_{fold}/X_test.npy")
    

    X_train3d = ImgRecon().makeRGB(data = X_train, frame = 330)
    X_test3d = ImgRecon().makeRGB(data = X_test, frame = 330)
    X_train3d, y_train = shuffle(X_train3d, np.array(y_train), random_state=42)

    y_trainC = tf.keras.utils.to_categorical(y_train, num_classes=nb_classes)
    y_testC = tf.keras.utils.to_categorical(y_test, num_classes=nb_classes)

    history = net.fit(X_train3d, y_trainC, epochs=nb_epoch, batch_size=64, validation_data=(X_test3d, y_testC))
    directory = f"SavedModels/C{nb_classes}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    net.save(os.path.join(directory,  ExpName + "Fold" + fold + ".keras"))

def extract_features(model, data):
    features = model.predict(data)
    return features
    
def modelImp(ExpName, foldId, nb_classes=5):
    fold_str = str(foldId + 1)
    Ftmodel = tf.keras.models.load_model(f"SavedModels/C{nb_classes}/" + ExpName + "Fold" + fold_str + ".keras")
    pretrained_layers = Ftmodel.layers[:-1]
    Ftmodel = Model(inputs=Ftmodel.input, outputs=pretrained_layers[-1].output)
    Ftmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return Ftmodel





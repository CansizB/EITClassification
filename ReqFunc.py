import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.models import Model
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

class ImgRecon:
    def __init__(self, path):
        self.path = path

    def makeRGB(self, data, frame):

        data3d = np.zeros((data.shape[0],data.shape[1],data.shape[2],data.shape[3],3))
        
        for i in range(frame):
            # Her bir frame'i 3 boyutlu hale getirme
            for j in range(3):
                data3d[:, i, :, :, j] = data[:, i, :, :]
        data3d = data3d.astype(np.uint8)
    
        return data3d


    def get_slice(self, CSVname):
        # Read and process the CSV data
        data = pd.read_csv(CSVname, header=None).values
        data[np.isnan(data)] = 0  # Replace NaNs with 0
        data[data == 0] = np.nan  # Convert 0s back to NaN
        min_val, max_val = np.nanmin(data), np.nanmax(data)
        normalized_data = (data - min_val) / (max_val - min_val) * 255

        # Slice images into chunks
        img_size = 32
        nb_img_vert = normalized_data.shape[0] // img_size
        imgs = [normalized_data[j * img_size: (j + 1) * img_size] for j in range(nb_img_vert)]
        return np.stack(imgs, axis=0)

    def RoiFrame(self, slice_data, start, end):
        return np.array([slice_data[i] for i in range(start, end)])

    def KFOLD(self, nb_subj, meta_path: str):

        xph=np.random.random(nb_subj)
        meta=pd.read_csv(meta_path + "/Metadata.txt")
        meta.reset_index(drop=True, inplace=True)
        yph=meta["Diagnosis"]
        label_encoder = LabelEncoder()
        yph = label_encoder.fit_transform(yph)
        
        # yph = np.where(yph == 0, 0, np.where(yph == 1, 0, np.where(yph == 2, 1, 2))) #if 3 class structure is implemented 
        # yph = np.where(yph == 2, 1, 0) #if binary structure is implemented 


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

    def CSVtoNumPy(self, class_info, meta, file_name, CSVname):
        train, test = [], []

        for i in range(0, 78):
            subject = i + 1
            if meta["Diagnosis"][i] == class_info:
                for j in range(len(file_name)):
                    if file_name["SubjectID"][j] == subject:
                        path = file_name["EITFilename"][j]
                        slice_data = self.get_slice(CSVname + "/" + path[:-4] + ".csv")
                        start, end = round(len(slice_data) / 2) - 165, round(len(slice_data) / 2) + 165
                        frames = self.RoiFrame(slice_data, start, end)

                        if subject in train:
                            train.append(frames)
                        if subject in test:
                            test.append(frames)

        return np.array(train), np.array(test)

    def GenSet(self, fold, path, save=False):
        # Assuming train/test indices are set from KFOLD
        train_idx, test_idx = self.train_test_splits[fold]
        CSV_name = path
    
        # Generate datasets for different classes
        COPD_train, COPD_test = self.CSVtoNumPy("COPD", meta, file_name, CSV_name)
        ILD_train, ILD_test = self.CSVtoNumPy("ILD", meta, file_name, CSV_name)
        Healthy_train, Healthy_test = self.CSVtoNumPy("Healthy", meta, file_name, CSV_name)
        Asthma_train, Asthma_test = self.CSVtoNumPy("Asthma", meta, file_name, CSV_name)
        Infection_train, Infection_test = self.CSVtoNumPy("Infection", meta, file_name, CSV_name)
    
        # Concatenate and label the data
        X_train = np.concatenate([COPD_train, ILD_train, Asthma_train, Infection_train, Healthy_train])
        X_test = np.concatenate([COPD_test, ILD_test, Asthma_test, Infection_test, Healthy_test])
        
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
    
        # Optionally save the arrays to .npy files
        if save:
            np.save(f"{path}/fold_{fold}/X_train.npy", X_train)
            np.save(f"{path}/fold_{fold}/X_test.npy", X_test)
            np.save(f"{path}/fold_{fold}/y_train.npy", y_train)
            np.save(f"{path}/fold_{fold}/y_test.npy", y_test)
    
        return X_train, X_test, y_train, y_test



def ftExp(net, fold, ExpName, nb_epoch, nb_classes):
    #for i in range(5):
    fold = str(i + 1)
    y_train = np.load(f"path/fold_{fold}/y_train.npy")
    y_test = np.load(f"path/fold_{fold}/y_test.npy")
    X_train = np.load(f"path/fold_{fold}/X_train.npy")
    X_test = np.load(f"path/fold_{fold}/X_test.npy")

    X_train3d = makeRGB(X_train, 330)
    X_test3d = makeRGB(X_test, 330)
    X_train3d, y_train = shuffle(X_train3d, np.array(y_train), random_state=42)

    y_trainC = tf.keras.utils.to_categorical(y_train, num_classes=nb_classes)
    y_testC = tf.keras.utils.to_categorical(y_test, num_classes=nb_classes)

    history = net.fit(X_train3d, y_trainC, epochs=nb_epoch, batch_size=64, validation_data=(X_test3d, y_testC))
    net.save(ExpName + "Fold" + fold + ".keras")

def extract_features(model, data):
    features = model.predict(data)
    return features
    
def modelImp(ExpName, foldId):
    fold_str = str(foldId)
    Ftmodel = tf.keras.models.load_model(ExpName + "Fold" + fold_str + ".keras")
    pretrained_layers = Ftmodel.layers[:-1]
    Ftmodel = Model(inputs=Ftmodel.input, outputs=pretrained_layers[-1].output)
    Ftmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return Ftmodel

def listdefinder(prefix):
    AccList = [[], [], []]
    bAccList = [[], [], []]
    F1List = [[], [], []]
    
    # Assign to global variables dynamically using prefix
    globals()[f"AccList_{prefix}"] = AccList
    globals()[f"bAccList_{prefix}"] = bAccList
    globals()[f"F1List_{prefix}"] = F1List

def create_proba_model_lists():
    model_names = ['RBF', 'LIN', 'HGBC', 'SoVC', 'RFC', 'SC']
    for model in model_names:
        globals()[f"proba_model{model}"] = []


def create_estimators():
    return [
        ('svmline', SVC(C= 1, gamma= "scale", kernel= "linear", degree=1, class_weight="balanced", probability=True)),
        ('svmrbf', SVC(C= 1, gamma= "scale", kernel= "rbf", degree=1, class_weight="balanced", probability=True)),
        ('hgbc', HistGradientBoostingClassifier(loss='log_loss', learning_rate= 0.1, max_iter=400, max_depth=4, class_weight="balanced")),
        ('rfc', RandomForestClassifier(n_estimators=200, max_depth=4, criterion="gini", class_weight="balanced"))
    ]





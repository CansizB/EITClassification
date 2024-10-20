import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix

import ReqFunc
from ReqFunc import ImgRecon
from EITExp import ModelTypes


def listdefinder(prefix):
    AccList = [[], [], []]
    bAccList = [[], [], []]
    F1List = [[], [], []]
    
    # Global değişken isimlerini dinamik olarak tanımla
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

def FitWorkflow(CSVname, mfile_path, EITname, model_name="densenet201", nb_classes=5):
    
    listdefinder('SVM_LIN')
    listdefinder('SVM_RBF')
    listdefinder('RFC')
    listdefinder('HGBC')
    listdefinder('SoVC')
    listdefinder('SC')
    listdefinder('HVC')


    create_proba_model_lists()

    estimators = create_estimators()
    for fold in range(5):
        #c= str(i+1)
        
        X_train, X_test, y_train, y_test = ImgRecon().GenSet(fold=fold, CSVname = CSVname, EITname=EITname, mfile_path = mfile_path, nb_classes= nb_classes, save=True)
    
        X_train3d= ImgRecon().makeRGB(data = X_train, frame = 330)
        X_test3d= ImgRecon().makeRGB(data = X_test,frame = 330)
    
        X_train3d, y_train = shuffle(X_train3d, np.array(y_train), random_state=42)

    
    
        for j in range(3):
    
            if j == 5:
                model = ModelTypes().IPWM(model_name= model_name)
                print("IPWM was utilized")
                
            
            elif j == 0:
                modelId = "FTM"
                model = ModelTypes().FTM(model_name= model_name, nb_classes= nb_classes)

                print("Fine-tuning of FTM is currently being utilized.") 
                ReqFunc.ftExp(net = model, fold = fold, ExpName = modelId, nb_epoch = 20, nb_classes= 5)
                model = ReqFunc.modelImp(ExpName = modelId, foldId=fold, nb_classes=nb_classes)
    
                print("Feature extraction is being processed by FTM") 
            
            else:
                modelId = "FTADLM"
                model = ModelTypes().FTM(model_name= model_name, nb_classes= nb_classes)

                print("Fine-tuning of FTADLM is currently being utilized.") 
                ReqFunc.ftExp(net = model, fold = fold, ExpName = modelId, nb_epoch = 10, nb_classes= 5)
                model = ReqFunc.modelImp(ExpName = modelId, foldId=fold, nb_classes=nb_classes)
    
                print("Feature extraction is being processed by FTADLM.") 
    
            feature_vectors_train = ReqFunc.extract_features(model, [X_train3d])
            feature_vectors_test = ReqFunc.extract_features(model, [X_test3d])
    
            sc = StandardScaler().fit(feature_vectors_train)
            X_trainSS = sc.transform(feature_vectors_train)
            X_testSS = sc.transform(feature_vectors_test)
    
            #SVM_LIN
            SVM_LIN= SVC(C= 1, gamma= "scale",kernel= "linear",degree=1,class_weight="balanced",probability= True)
            SVM_LIN.fit(X_trainSS,np.ravel(y_train))
            y_predSVM_LIN = SVM_LIN.predict(X_testSS)
            conf_matSVM_LIN = confusion_matrix(y_test, y_predSVM_LIN)
            acc = accuracy_score(y_test, y_predSVM_LIN)
            f1= f1_score(y_test, y_predSVM_LIN, average='macro')
            bAcc= balanced_accuracy_score(y_test, y_predSVM_LIN)
    
            AccList_SVM_LIN[j].append(acc)
            F1List_SVM_LIN[j].append(f1)
            bAccList_SVM_LIN[j].append(bAcc)
    
            #SVM_RBF
            SVM_RBF= SVC(C= 1, gamma= "scale",kernel= "rbf",degree=1,class_weight="balanced",probability= True)
            SVM_RBF.fit(X_trainSS,np.ravel(y_train))
            y_predSVM_RBF = SVM_RBF.predict(X_testSS)
            conf_matSVM_RBF = confusion_matrix(y_test, y_predSVM_RBF)
            acc= accuracy_score(y_test, y_predSVM_RBF)
            f1=f1_score(y_test, y_predSVM_RBF, average='macro')
            bAcc= balanced_accuracy_score(y_test, y_predSVM_RBF)
    
            AccList_SVM_RBF[j].append(acc)
            F1List_SVM_RBF[j].append(f1)
            bAccList_SVM_RBF[j].append(bAcc)
    
            #HGBC
            HGBC= HistGradientBoostingClassifier(loss='log_loss', learning_rate= 0.1, max_iter= 400, max_depth=4,class_weight="balanced")
            HGBC.fit(X_trainSS,np.ravel(y_train))
            y_predHGBC = HGBC.predict(X_testSS)
            acc= accuracy_score(y_test, y_predHGBC)
            f1=f1_score(y_test, y_predHGBC, average='macro')
            bAcc= balanced_accuracy_score(y_test, y_predHGBC)
            
            AccList_HGBC[j].append(acc)
            F1List_HGBC[j].append(f1)
            bAccList_HGBC[j].append(bAcc)
    
            #RFC
            RFC= RandomForestClassifier(n_estimators=200,max_depth=4,criterion="gini",class_weight="balanced")
            RFC.fit(X_trainSS,np.ravel(y_train))
            y_predRFC=RFC.predict(X_testSS)
            conf_matRFC = confusion_matrix(y_test, y_predRFC)
            acc = accuracy_score(y_test, y_predRFC)
            f1= f1_score(y_test, y_predRFC, average='macro')
            bAcc= balanced_accuracy_score(y_test, y_predRFC)
    
            AccList_RFC[j].append(acc)
            F1List_RFC[j].append(f1)
            bAccList_RFC[j].append(bAcc)
    
            #Stacking
            SC = StackingClassifier(estimators=estimators, cv=None)
            SC.fit(X_trainSS,np.ravel(y_train))
            y_predSC=SC.predict(X_testSS)
            f1=f1_score(y_test,y_predSC,average="macro")
            acc=accuracy_score(y_test,y_predSC)
            bAcc= balanced_accuracy_score(y_test, y_predSC)
    
            AccList_SC[j].append(acc)
            F1List_SC[j].append(f1)
            bAccList_SC[j].append(bAcc)
    
            #SoVoting
            SoVC=VotingClassifier(estimators=estimators,voting="soft")
            SoVC.fit(X_trainSS,np.ravel(y_train))
            y_predSoVC=SoVC.predict(X_testSS)
            f1=f1_score(y_test,y_predSoVC,average="macro")
            acc=accuracy_score(y_test,y_predSoVC)
            bAcc= balanced_accuracy_score(y_test, y_predSoVC)
    
            AccList_SoVC[j].append(acc)
            F1List_SoVC[j].append(f1)
            bAccList_SoVC[j].append(bAcc)
    
            #HVoting
            HVC=VotingClassifier(estimators=estimators,voting="hard")
            HVC.fit(X_trainSS,np.ravel(y_train))
            y_predHVC=HVC.predict(X_testSS)
            f1=f1_score(y_test,y_predHVC,average="macro")
            acc=accuracy_score(y_test,y_predHVC)
            bAcc= balanced_accuracy_score(y_test, y_predHVC)
    
            AccList_HVC[j].append(acc)
            F1List_HVC[j].append(f1)
            bAccList_HVC[j].append(bAcc)
    
    
            proba_modelRBF.append(SVM_RBF.predict_proba(X_testSS))
            proba_modelLIN.append(SVM_LIN.predict_proba(X_testSS)) 
            proba_modelRFC.append(RFC.predict_proba(X_testSS)) 
            proba_modelHGBC.append(HGBC.predict_proba(X_testSS)) 
            proba_modelSoVC.append(SoVC.predict_proba(X_testSS))
            proba_modelSC.append(SC.predict_proba(X_testSS))

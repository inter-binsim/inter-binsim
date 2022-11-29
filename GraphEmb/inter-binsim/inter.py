import xgboost as xgb
from glob import glob
import sys
import pickle
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeCV
from pandas.core.frame import DataFrame
import shap
def x_pre():
    dir_n = 'feature-train/*'
    dir_l = glob (dir_n)

    params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 2,
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
    }
    plst = list(params.items())
    num_rounds = 500
    c_ = 0
    for item in dir_l:
        if c_<10:
            c_+=1
            continue
        c_+=1
        with open(item ,'rb') as f: # [IO]
            print('item:', item)
            t = pickle.load(f)
            #print('t:', t)
            data_a = []
            label_a = []
            for data in t:
                data_a.append(data[2])
                label_a.append(data[3])
            x_train, x_test, y_train, y_test = train_test_split(data_a, label_a, test_size=0.2, random_state=1234565)
            dtrain = xgb.DMatrix(x_train, y_train)
            #print('x_train.shape:', x_train[0], len(x_train), dtrain.shape)
            model = xgb.train(plst, dtrain, num_rounds)
            dtest = xgb.DMatrix(x_test)
            ans2 = model.predict(dtest)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(dtrain)
            print('s v:', shap_values[0].shape, len(shap_values))
            x_train = DataFrame(x_train)
            shap.summary_plot(shap_values, x_train)
            plt.savefig('scratch.png')

            #gm = GaussianMixture(n_components=2, covariance_type = 'full',  tol = 0.5, n_init=8, max_iter=1000)
            #gm = LinearSVC()
            #gm = RidgeCV()
            #gm.fit(x_train, y_train)
            #ans =  gm.predict(x_test)
            #print('ans:', ans)
            #ans[ans>0.5]=1
            #ans[ans<=0.5]=0
            
            #print('gm.coef_:', gm.coef_)
            
            # print('tmp:', tmp)
            cnt1 = 0
            cnt2 = 0
            cnt3 = 0
            cnt4 = 0
            for i in range(len(y_test)):
                # if ans[i] == y_test[i]:
                #     cnt1 += 1
                # else:
                #     cnt2 += 1
                if ans2[i] == y_test[i]:
                    cnt3 += 1
                else:
                    cnt4 +=1

            #print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))
            print("xboost Accuracy: %.2f %% " % (100 * cnt3 / (cnt3 + cnt4)))
            plot_importance(model, importance_type='total_gain')
            plt.savefig('./fea_importance.png')
            plot_importance(model, importance_type='gain')
            plt.savefig('./fea_importance_2.png')
            
            #plt.show()
            #print('x_train:', len(x_train), len(data_a))
            #print('data a:', data_a)
            #dtrain = xgb.DMatrix( data_a, label=label_a)
            #print('t:', t[-1])
            sys.exit()
            
if __name__ == '__main__':
    x_pre()
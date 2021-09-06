from flask import Flask, render_template, request, url_for
import pickle
from dnn_app_utils_v3 import *
import numpy as np
from PIL import Image
import pandas as pd

filename = 'parameters.sav'
filename2 = 'titanic_model_voting.sav'
filename3 = 'parkinsons_model.sav'
loaded_parameters = pickle.load(open(filename, 'rb'))
titanic_model = pickle.load(open(filename2, 'rb'))
parkinsons_model = pickle.load(open(filename3, 'rb'))
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

def predict(X, parameters):
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    return p

app = Flask(__name__, template_folder='template')

@app.route('/') 
def index():
    return render_template('index.html')



@app.route("/prediction", methods = ['POST'])
def cat_prediction():

    img = request.files['img']
    img.save("static/img.jpg")
    num_px = 64
    # my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
    fname = "static/" + "img.jpg"
    image = np.array(Image.open(fname).resize((num_px, num_px)))
    plt.imshow(image)
    image = image / 255.
    image = image.reshape((1, num_px * num_px * 3)).T
    my_predicted_image = predict(image, loaded_parameters)
    result = classes[int(np.squeeze(my_predicted_image)),].decode("utf-8")
    return render_template("cat_prediction.html", data= result)

def survival(x):
    if x == 0.0:
        temp = "Not Survive"
    else:
        temp = "Survived"
    return temp
@app.route("/titanic_prediction", methods = ['POST'])
def titanic_prediction():
    data1 = request.form['Ticket Class']
    data2 = request.form['Gender']
    data3 = request.form['Age']
    data4 = request.form['sibsp']
    data5 = request.form['parch']
    data6 = request.form['Passenger fare']
    data7 = request.form['Port of Embarkation']
    observation = np.array([[data1, data2, data3, data4, data5, data6, data7]])
    titanic_predict = titanic_model.predict(observation)
    titanic_result = survival(titanic_predict[0])
    return render_template("titanic_pred.html", data= titanic_result)

@app.route("/parkinson_prediction", methods = ['POST'])
def parkinson_prediction():
    park1 = request.form['park_age']
    park2 = request.form['park_sex']
    park3 = request.form['jitter']
    park4 = request.form['shimmer']
    park5 = request.form['NHR']
    park6 = request.form['HNR']
    park7 = request.form['RPDE']
    park8 = request.form['DFA']
    park9 = request.form['PPE']
    park_observation = np.array([[park1, park2, park3, park4, park5, park6, park7, park8, park9]])
    parkinson_predict = parkinsons_model.predict(park_observation)
    park_result = parkinson_predict[0]
    return render_template("park_pred.html", data= park_result)

def iqr_cal(q1, q3):
    iqr = q3 - q1
    s = 1.5* iqr
    UB = s +q3
    LB = q1 - s
    return q1, q3, iqr, s, UB, LB
def empirical_rule_cal(mean, std_dev):
    one_std = []
    two_std = []
    three_std = []
    one_std.extend([mean-std_dev, mean+std_dev])
    two_std.extend([mean-(2*std_dev), mean+(2*std_dev)])
    three_std.extend([mean-(3*std_dev), mean+(3*std_dev)])
    return one_std, two_std, three_std
def z_score_cal(data_point, mean, std_dev):
    z_score = (data_point - mean) /std_dev
    return data_point, mean, std_dev, z_score
@app.route('/calculator') 
def calculator():
    return render_template('cal.html')

@app.route("/calculator", methods = ['POST'])
def iqr():
    qua1 = request.form['q1']
    qua3 = request.form['q3']
    q1, q3, iqr, s, UB, LB = iqr_cal(float(qua1), float(qua3))
    iqr_result = [q1, q3, iqr, s, UB, LB]
    return render_template("cal_iqr.html", q1 =q1 , q3=q3, iqr=iqr, s=s, UB=UB, LB=LB)

@app.route("/empirical", methods = ['POST'])
def empirical():
    mean = request.form['mean_emp']
    std = request.form['std_emp']
    one_std, two_std, three_std = empirical_rule_cal(float(mean), float(std))
    return render_template("cal_emp.html", one_std=one_std, two_std=two_std, three_std=three_std )

@app.route("/z_score", methods = ['POST'])
def z_score():
    data_point_z = request.form['data_point_z']
    mean_z = request.form['mean_z']
    std_z = request.form['std_dev_z']
    data_point, mean, std_dev, z_score = z_score_cal(float(data_point_z), float(mean_z), float(std_z))
    return render_template("cal_z.html", data_point=data_point, mean=mean, std_dev=std_dev, z_score=z_score)

def deterministic(d, h, K):
    Q = np.sqrt((2*d*K)/h)
    t = Q/d
    return d, h, K, Q, t

@app.route("/deterministic", methods = ['POST'])
def deterministic_continuous():
    d_dcr = request.form['d_dcr']
    h_dcr = request.form['h_dcr']
    K_dcr = request.form['K_dcr']
    d, h, K, Q, t = deterministic(float(d_dcr), float(h_dcr), float(K_dcr))
    return render_template("cal_dcrm.html", d=d, h=h, K=K, Q=Q, t=t)

def deterministic_shortage(d, h, K, p):
    S = (np.sqrt((2*d*K)/h))*(np.sqrt(p/(p+h)))
    Q = (np.sqrt((2*d*K)/h))*(np.sqrt((p+h)/p))
    t = Q/d
    Q_S = Q-S
    return d, h, K, p, S, Q, t, Q_S

@app.route("/deterministic_shortage", methods = ['POST'])
def shortage():
    d_sa = request.form['d_sa']
    h_sa = request.form['h_sa']
    K_sa = request.form['K_sa']
    p_sa = request.form['p_sa']
    d, h, K, p, S, Q, t, Q_S = deterministic_shortage(float(d_sa), float(h_sa), float(K_sa), float(p_sa))
    return render_template("cal_dcrm_sa.html", d=d, h=h, K=K, p=p, S=S, Q=Q, t=t, Q_S=Q_S)

from scipy.stats import norm
def stochastic_continuous(Q, L, mean, std):
    ZL = norm.ppf(0.95)
    ss = ZL *std
    R = mean + ss
    return Q, L, mean, std, ZL, ss, R

@app.route("/stochastic", methods = ['POST'])
def stochastic():
    q_st = request.form['q_st']
    l_st = request.form['l_st']
    mean_st = request.form['mean_st']
    std_st = request.form['std_st']
    Q, L, mean, std, ZL, ss, R = stochastic_continuous(float(q_st), float(l_st), float(mean_st), float(std_st))
    return render_template("cal_stcrm_sa.html", Q=Q, L=L, mean=mean, std=std, ZL=ZL, ss=ss,R=R)

@app.route('/confusion_matrix') 
def confusion_matrix():
    return render_template('confusion.html')

def confusion(tp, fn, fp, tn, total1, total2, n):
    accuracy = (tp+tn)/n
    precision = (tp/(tp+fp))
    recall = tp/(tp+fn)
    f1 = (2*precision*recall)/(precision+recall)
    misclassification = (fp+fn)/n
    tpr = tp/total1
    fpr = fp/total2
    specificity = tn/total2
    return accuracy, precision, recall, f1, misclassification, tpr, fpr, specificity

@app.route("/confusion_matrix", methods = ['POST'])
def confusion_matrix_cal():
    tp = request.form['tp']
    fn = request.form['fn']
    fp = request.form['fp']
    tn = request.form['tn']
    tp = int(tp)
    fn = int(fn)
    fp = int(fp)
    tn = int(tn)
    n = tp+fn+fp+tn
    total1 = (tp+fn)
    total2 = (fp+tn)
    total3 = (tp+fp)
    total4 = (fn+tn)
    accuracy, precision, recall, f1, misclassification, tpr, fpr, specificity = confusion(tp, fn, fp, tn, total1, total2, n)
    return render_template("confusion_result.html", tp=tp,fn=fn, fp=fp, tn=tn,n=n,total1=total1,total2=total2, total3=total3,total4=total4, accuracy=accuracy,precision=precision, recall=recall, f1=f1, misclassification=misclassification, tpr=tpr, fpr=fpr, specificity=specificity)

@app.route('/magic_box') 
def standard_deviation():
    return render_template('magic_box.html')

import statistics
@app.route("/standard_deviation", methods = ['POST'])
def standard_deviation_cal():
    values = request.form['my_std']
    values = values.split(",")
    float_values = [float(x) for x in values]
    mean = statistics.mean(float_values)
    stdev = statistics.stdev(float_values)
    fmean = statistics.fmean(float_values)
    harmonic_mean = statistics.harmonic_mean(float_values)
    median = statistics.median(float_values)
    median_low = statistics.median_low(float_values)
    median_high = statistics.median_high(float_values)
    quantiles = statistics.quantiles(float_values)
    pstdev = statistics.pstdev(float_values)
    pvariance = statistics.pvariance(float_values)
    variance = statistics.variance(float_values)
    mode = statistics.mode(float_values)
    multimode = statistics.multimode(float_values)
    return render_template("magic_box.html", mode=mode, multimode=multimode,mean = mean, stdev = stdev, fmean = fmean, harmonic_mean =harmonic_mean, median_low = median_low, median_high = median_high , median=median,quantiles=quantiles,pstdev=pstdev,pvariance=pvariance, variance=variance)
if __name__ == "__main__":
    app.run(debug=True)


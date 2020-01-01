import os
from six.moves import cPickle as pickle
import numpy as np
import scipy.io.wavfile
from sklearn.mixture import GaussianMixture as GMM
from python_speech_features import mfcc
from python_speech_features import delta
from sklearn import preprocessing
import warnings
import sys

warnings.filterwarnings("ignore")

path = os.path.dirname(os.path.abspath(__file__))
source = path + '\\trainall\\'

train_files_males = [os.path.join(source, f) for f in os.listdir(source) if f.endswith('M.wav')]
train_files_females = [os.path.join(source, f) for f in os.listdir(source) if f.endswith('K.wav')]

# Making model=================================================

def extractfeatures(path):
        sr, audio = scipy.io.wavfile.read(path)
        mfcc_feature = mfcc(
            audio,
            sr,  # sample rate of the signal
            winlen=0.05,  # length of the analysis window in seconds (default 0.025s)
            winstep=0.01,  # step between successive windows in seconds (default 0.01s)
            numcep=13,  # number of cepstrum to return (default 13)
            nfilt=30,  # number of filters in the filterbank (default 26)
            nfft=4096,  # FFT size (default 512)
            appendEnergy=True)

        mfcc_feature = preprocessing.scale(mfcc_feature)
        deltas = delta(mfcc_feature, 2)
        double_deltas = delta(deltas, 2)
        combined = np.hstack((mfcc_feature, deltas, double_deltas))
        return combined

def collect_features(files):
    features = np.asarray(())
    for file in files:
        vector = extractfeatures(file)
        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))
    return features

def save_gmm(gmm, name):
    filename = name + ".gmm"
    with open(filename, 'wb') as gmm_file:
        pickle.dump(gmm, gmm_file)
    print("SAVING", filename)

def get_models():
    if os.path.exists("females.gmm"):
        os.remove("females.gmm")
    if os.path.exists("males.gmm"):
        os.remove("males.gmm")
    female_features = collect_features(train_files_females)
    male_features = collect_features(train_files_males)

    females_gmm = GMM(n_components=16, max_iter=200, covariance_type='diag', n_init=3)
    males_gmm = GMM(n_components=16, max_iter=200, covariance_type='diag', n_init=3)

    females_gmm.fit(female_features)
    males_gmm.fit(male_features)

    save_gmm(females_gmm, "females")
    save_gmm(males_gmm, "males")

# Testing data=========================================

# test_data = path + '\\test_data\\K\\'
# test_data = path + '\\próbki\\' #moja próbka
# test_files = [os.path.join(test_data, f) for f in os.listdir(test_data)]


file = sys.argv[1]
filepath = path + '\\'
test_files = [os.path.join(filepath, file)]


def choose_gender(vector):
    females_gmm = pickle.load(open('females.gmm', 'rb'))
    males_gmm = pickle.load(open('males.gmm', 'rb'))
    # female hypothesis scoring
    is_female_scores = np.array(females_gmm.score(vector))
    is_female_log_likelihood = is_female_scores.sum()
    # male hypothesis scoring
    is_male_scores = np.array(males_gmm.score(vector))
    is_male_log_likelihood = is_male_scores.sum()

    # print("FEMALE SCORE: ", str(round(is_female_log_likelihood, 3)))
    # print("MALE SCORE: ", str(round(is_male_log_likelihood, 3)))

    if is_male_log_likelihood > is_female_log_likelihood:
        gender = "M"
    else:
        gender = "K"
    return gender

def recognise(files):
    # MALES = 0
    # FEMALES = 0
    for file in files:
        # print("TESTING: ", os.path.basename(file))
        vector = extractfeatures(file)
        gender = choose_gender(vector)
        print(gender)
        # if gender == "male":
        #    MALES += 1
        # else:
        #    FEMALES += 1
    # print("RECOGNISED FEMALES: ", FEMALES)
    # print("RECOGNISED MALES: ", MALES)
    # print("FILES:", len(test_files))


#get_models()
recognise(test_files)



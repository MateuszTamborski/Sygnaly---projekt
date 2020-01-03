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

# ======== Making model====================
def gather_features(filepath):
        sr, signal = scipy.io.wavfile.read(filepath)
        mfcc_feature = mfcc(
            signal,
            sr,  # sample rate of the signal
            winlen=0.05,  # length of the analysis window in seconds (default 0.025s)
            winstep=0.01,  # step between successive windows in seconds (default 0.01s)
            numcep=13,  # number of cepstrum to return (default 13)
            nfilt=30,  # number of filters in the filterbank (default 26)
            nfft=4096,  # FFT size (default 512)
            appendEnergy=True)

        mfcc_feature = preprocessing.scale(mfcc_feature)
        deltas = delta(mfcc_feature, 2)
        delta_deltas = delta(deltas, 2)
        stack = np.hstack((mfcc_feature, deltas, delta_deltas))
        return stack

def collect_features(files):
    features = np.asarray(())
    for file in files:
        vector = gather_features(file)
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

    females_gmm = GMM(n_components=22, covariance_type='diag', max_iter=200, n_init=3)
    males_gmm = GMM(n_components=22, covariance_type='diag', max_iter=200, n_init=3)

    females_gmm.fit(female_features)
    males_gmm.fit(male_features)

    save_gmm(females_gmm, "females")
    save_gmm(males_gmm, "males")


# ==============Testing data========================
filepath = sys.argv[1]
test_files = [os.path.join(path, filepath)]

def choose_gender(vector):
    females_gmm = pickle.load(open('females.gmm', 'rb'))
    males_gmm = pickle.load(open('males.gmm', 'rb'))

    female_score = females_gmm.score(vector)
    male_score = males_gmm.score(vector)

    if male_score > female_score:
        gender = "M"
    else:
        gender = "K"
    return gender

def recognise(files):
    for file in files:
        vector = gather_features(file)
        gender = choose_gender(vector)
        print(gender)


if __name__ == "__main__":
    # get_models()            # makes models from files in trainall folder
    recognise(test_files)   # recognises gender from the given file

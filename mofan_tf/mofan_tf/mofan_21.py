import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit.transform(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

def add_layer(inputs,in_size,out_side,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_side]))
    biases = tf.Variable(tf.zeros([1,out_side]))+0.1
    Wx_plus_b = tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


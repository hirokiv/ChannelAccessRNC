import mglearn
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np

def SVM_regression(x,y):

  try:

    amp = max(y) - min(y)

    y = y / amp
  #  # データを用意する------------------------------------------------------------------
  #  x = np.arange(0.0, 10.0, 0.1)                           # 横軸を作成
  #  noise = np.random.normal(loc=0, scale=0.1, size=len(x)) # ガウシアンノイズを生成
  #  y = np.sin(2 * np.pi * 0.2 * x) + noise                 # 学習用サンプル波形
  #  # ----------------------------------------------------------------------------------

    #  learn by svm
    model = svm.SVR(C=0.1, kernel='rbf', epsilon=0.01, gamma='auto')    # RBF kernel
    model.fit(x.reshape(-1, 1), y)                          # fitting

    # predict with supervised parameter set
    x_reg = x                         # generate x axis for regression
    y_reg = model.predict(x_reg.reshape(-1, 1))             # predicted
    r2 = model.score(x.reshape(-1, 1), y)                   # determine coefficient

    # Plot configuration---------------------------------------------------------------
    plt.rcParams['font.size'] = 14

    # show scales internally
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    # add measures in each side of the graph
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # データプロットの準備。
    ax1.scatter(x, y, label='Dataset', lw=1, marker="o")
    ax1.plot(x_reg, y_reg, label='Regression curve', color='red')
    plt.legend(loc='upper right')

    # グラフ内に決定係数を記入
    plt.text(0.0, -0.5, '$\ R^{2}=$' + str(round(r2, 2)), fontsize=20)
    fig.tight_layout()
    plt.show()
    fig.savefig('../Image/' + 'regression_result.png')

    plt.clf()
    plt.close(fig)

    error_flag = 0 # no error
    return y_reg * amp, amp, error_flag

#  except FloatingPointError | ZeroDivisionError as e:
  except FloatingPointError  as error_flag:
    print('SVM_regression false' )
    return np.zeros_like(y), 0, error_flag
    print(e)



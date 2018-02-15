import matplotlib.pyplot as plot
import numpy             as np
import pickle

from data_set     import load_data
from funcy        import last, partial
from keras.models import load_model
from operator     import getitem
from utility      import ZeroPadding


labels = ['M・A・O:若狭悠里',
          '井口裕香:三宅日向',
          '佐藤聡美:田井中律',
          '原紗友里:大垣千明',
          '子安武人:ロズワール・L・メイザース',
          '寿美菜子:琴吹紬',
          '小林裕介:ナツキ・スバル',
          '小澤亜李:恵飛須沢胡',
          '山寺宏一:スパイク・スピーゲル',
          '新井里美:ベアトリス',
          '日笠陽子:秋山澪',
          '日野由利加:カテリーナ',
          '村川梨衣:ラム',
          '東山奈央:志摩リン',
          '水瀬いのり:レム',
          '水瀬いのり:丈槍由紀',
          '水瀬いのり:玉木マリ',
          '石塚運昇:ジェット・ブラック',
          '福島潤:佐藤和真',
          '竹達彩奈:中野梓',
          '花守ゆみり:各務原なでしこ',
          '花澤香菜:小淵沢報瀬',
          '茅野愛衣:佐倉慈',
          '西凛太朗:アシモフ・ソーレンサン',
          '豊崎愛生:ゆんゆん',
          '豊崎愛生:平沢唯',
          '豊崎愛生:犬山あおい',
          '雨宮天:アクア',
          '高橋李衣:エミリア',
          '高橋李衣:めぐみん',
          '高橋李衣:斉藤恵那',
          '高橋李衣:直樹美紀']


def plot_history():
    def plot_values_collection(title, values_collection):
        plot.clf()
        plot.title(title)
        for values in values_collection:
            plot.plot(values)
        plot.show()

    with open('./results/history.pickle', 'rb') as f:
        history = pickle.load(f)

    print(last(history['val_acc']))

    plot_values_collection('loss',     map(partial(getitem, history), ('loss', 'val_loss')))
    plot_values_collection('accuracy', map(partial(getitem, history), ('acc',  'val_acc')))


def print_predicts():
    _, (x, y) = load_data()
    x = (x - 4.3854903e-05) / 0.042366702

    model  = load_model('./results/model.h5', custom_objects={'ZeroPadding': ZeroPadding})
    y_pred = np.argmax(model.predict(x), axis=1)

    for t, p in zip(y, y_pred):
        print('{0}\t{1}'.format(labels[t], labels[p]))

    del model


def main():
    plot_history()
    print_predicts()


if __name__ == '__main__':
    main()

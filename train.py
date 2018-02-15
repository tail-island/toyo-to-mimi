import numpy as np
import pickle

from data_set           import data_generator, load_data
from funcy              import identity, juxt, partial, rcompose, repeatedly
from keras.callbacks    import ReduceLROnPlateau
from keras.layers       import *
from keras.models       import Model, save_model
from keras.regularizers import l2
from utility            import ZeroPadding


def computational_graph(class_size):
    # Utilities

    def ljuxt(*fs):
        return rcompose(juxt(*fs), list)

    def add():
        return Add()

    def batch_normalization():
        return BatchNormalization()

    def conv(filters, kernel_size):
        return Conv1D(filters, kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0001), use_bias=False)

    def dense(units):
        return Dense(units, kernel_regularizer=l2(0.0001))

    def global_average_pooling():
        return GlobalAveragePooling1D()

    def max_pooling():
        return MaxPooling1D()

    def relu():
        return Activation('relu')

    def softmax():
        return Activation('softmax')

    def zero_padding(filter_size):
        return ZeroPadding(filter_size)

    # Computational graph.

    def wide_residual_net():
        def residual_unit(filter_size):
            return rcompose(ljuxt(rcompose(batch_normalization(),
                                           conv(filter_size, 3),
                                           batch_normalization(),
                                           relu(),
                                           conv(filter_size, 3),
                                           batch_normalization()),
                                  identity),
                            add())

        def residual_block(filter_size, unit_size):
            return rcompose(zero_padding(filter_size),
                            rcompose(*repeatedly(partial(residual_unit, filter_size), unit_size)))

        return rcompose(conv(160, 3),
                        max_pooling(),
                        residual_block(160, 1),
                        max_pooling(),
                        residual_block(160, 1),
                        max_pooling(),
                        residual_block(160, 1),
                        max_pooling(),
                        residual_block(160, 1),
                        max_pooling(),
                        residual_block(320, 1),
                        max_pooling(),
                        residual_block(320, 1),
                        max_pooling(),
                        residual_block(320, 2),
                        max_pooling(),
                        residual_block(640, 4))

    return rcompose(wide_residual_net(),
                    conv(256, 1),
                    global_average_pooling(),
                    dense(class_size),
                    softmax())


def main():
    (waves, labels), (x_validate, y_validate) = load_data()

    x_mean = 4.3854903e-05  # np.concatenate(waves).mean()
    x_std  = 0.042366702    # np.concatenate(waves).std()

    waves = tuple(map(lambda wave: (wave - x_mean) / x_std, waves))
    x_validate = (x_validate - x_mean) / x_std

    model = Model(*juxt(identity, computational_graph(max(y_validate) + 1))(Input(shape=x_validate.shape[1:])))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    batch_size =  50
    epoch_size = 500

    results = model.fit_generator(data_generator(waves, labels, batch_size), steps_per_epoch=8000 // batch_size, epochs=epoch_size,
                                  validation_data=(x_validate, y_validate),
                                  callbacks=[ReduceLROnPlateau(factor=0.5, patience=50, verbose=1)])

    with open('./results/history.pickle', 'wb') as f:
        pickle.dump(results.history, f)

    save_model(model, './results/model.h5')

    del model


if __name__ == '__main__':
    main()

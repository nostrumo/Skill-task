import pandas as pd
import numpy as np
import sources as src
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint



def model_baseline():
    from keras import Sequential
    from keras.layers import Dense
    from keras.models import Model
    from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, Input

    classifier = Sequential()
    # First Hidden Layer
    classifier.add(Dense(16, activation='relu', kernel_initializer='random_normal', input_dim=data_train.shape[1]))
    # Second  Hidden Layer
    classifier.add(Dense(8, activation='relu', kernel_initializer='random_normal'))
    classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
    # Output Layer
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    classifier.summary()
    return classifier


# Функция деления выборки на тестовую и тренеровочную
def data_spliter(dataset,y_dataset,param=0.2):
    data_test, data_train, y_data_test, y_data_train=train_test_split(dataset, y_dataset, test_size=param)

    print('data_test length :',data_test.shape)
    print('data_train length :',data_train.shape)
    print('y_data_test length :',y_data_test.shape)
    print('y_data_train length :',y_data_train.shape)

    return data_test, data_train, y_data_test, y_data_train


# Функция открываеющая необходимые данные и возвращяющая проедобработанные данные.
def data_processing(data_path, train=False):
    data_frame = pd.read_csv(data_path,sep=',',index_col='sample_id')   # Открываем данные
    if train:
        y_frame = data_frame['y']
        data_frame = data_frame.drop('y', 1)

    data_frame = data_frame.fillna(0)   # Убираем все значения Nan и меняем их на 0
    data_frame_norm = (data_frame - data_frame.mean()) / (data_frame.max() - data_frame.min())  # Нормируем данные

    if train:
        return data_frame_norm, y_frame
    else:
        return data_frame_norm


# Подготавливаем данные для нейросети
x_train_data, y_train_data = data_processing(src.train_data, train=True)
x_test_data = data_processing(src.test_data, train=False)

# Поделим нашу тренировочную выборку на тренировочную и тестовую ( 80% и 20% соостветственно )
data_test, data_train, y_data_test, y_data_train = data_spliter(x_train_data, y_train_data)

# Обучаем модель
model = model_baseline()
history = model.fit(x=data_train,
                    y=y_data_train,
                    validation_data=(data_test,y_data_test),
                    epochs=100,
                    batch_size=32)

# Строим графики loss и accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



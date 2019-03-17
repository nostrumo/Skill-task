import pandas as pd
import numpy as np
import sources as src
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint



def model_baseline():
    import keras.optimizers as optimizers
    from keras import Sequential
    # opt = SGD(lr=0.01)
    from keras.layers import Dense
    from keras.models import Model
    from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, Input

    Input_figure = Input(shape=(data_train.shape[1],), name='input1')

    x = Dense(8, activation='relu')(Input_figure)
    x = Dropout(0.2)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=Input_figure, outputs=out)
    model.summary()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=["accuracy"]
                  )
    return model


# Функция деления выборки на тестовую и тренеровочную
def data_spliter(dataset,y_dataset,param=0.3):
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
print(x_test_data)
# Поделим нашу тренировочную выборку на тренировочную и тестовую ( 80% и 20% соостветственно )
data_train, data_test, y_data_train, y_data_test = data_spliter(x_train_data, y_train_data,0.2)

# Обучаем модель
model = model_baseline()
history = model.fit(x=data_train,
                    y=y_data_train,
                    validation_data=(data_test,y_data_test),
                    epochs=20,
                    shuffle=True,
                    batch_size=5)

result=model.evaluate(data_test,y_data_test)
print(result)

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
model.save('my_model1.h5')

from keras.models import load_model

model = load_model('my_model1.h5')
prediction=[]
for row in np.array(x_test_data):
    prediction.append(model.predict(np.array([row])))
samples_ids=x_test_data.index.values.tolist()
print(prediction)
with open('my_csv.csv', 'w') as f:
    f.write('sample_id,y\n')
    for i in range(len(prediction)):
        f.write(f'{samples_ids[i]},{prediction[i][0][0]}\n')

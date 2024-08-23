from keras import Input
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Activation, AveragePooling2D, Flatten, Concatenate, \
    MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.models import Model


# Model č.1
def cnn_3x64_max_pooling():

    # Image input branch with 3 convolutional layers
    image_input = Input(shape=(480, 640, 3))
    x = Conv2D(64, (3, 3), padding='same')(image_input)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(x)

    x = Flatten()(x)

    # Numerical input branch for distance, angle, speed
    numerical_input = Input(shape=(3,))
    y = Dense(16, activation='relu')(numerical_input)

    # Concatenate the two branches
    combined = Concatenate()([x, y])

    # Adding a fully connected layer on top of the concatenated outputs
    z = Dense(64, activation='relu')(combined)
    z = Dense(32, activation='relu')(z)
    output = Dense(5, activation='linear')(z)  # 5 outputs based on the actions

    # Create the model
    model = Model(inputs=[image_input, numerical_input], outputs=output)

    # Compiling the model
    model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    return model


# Model č.2
def cnn_4_layers_max_pooling():
    image_input = Input(shape=(480, 640, 3))
    x = Conv2D(16, (3, 3), padding='same')(image_input)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(x)

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = Flatten()(x)

    # Numerical input branch for the distance and angle
    numerical_input = Input(shape=(3,))  # Assuming 3 inputs for angle, distance and speed
    y = Dense(16, activation='relu')(numerical_input)

    # Concatenate the two branches
    combined = Concatenate()([x, y])

    # Adding a fully connected layer on top of the concatenated outputs
    z = Dense(64, activation='relu')(combined)
    z = Dense(32, activation='relu')(z)
    # 5 outputs based on the actions
    output = Dense(5, activation='linear')(z)

    # Create the model
    model = Model(inputs=[image_input, numerical_input], outputs=output)

    # Compiling the model
    model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    return model


# Model č.3
def cnn_5_layers_max_pooling():
    image_input = Input(shape=(480, 640, 3))
    x = Conv2D(16, (3, 3), padding='same')(image_input)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(x)

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = Flatten()(x)

    # Numerical input branch for the distance and angle
    numerical_input = Input(shape=(3,))  # Assuming 2 inputs for angle and distance and speed
    y = Dense(16, activation='relu')(numerical_input)
    y = Dense(32, activation='relu')(y)
    # Concatenate the two branches
    combined = Concatenate()([x, y])

    # Adding a fully connected layer on top of the concatenated outputs
    z = Dense(128, activation='relu')(combined)
    z = Dense(64, activation='relu')(z)
    z = Dense(32, activation='relu')(z)
    z = Dense(16, activation='relu')(z)
    # 5 outputs based on the actions
    output = Dense(5, activation='linear')(z)

    # Create the model
    model = Model(inputs=[image_input, numerical_input], outputs=output)

    # Compiling the model
    model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    return model




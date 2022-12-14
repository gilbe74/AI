from keras.layers import PReLU, RepeatVector, TimeDistributed, GRU, ConvLSTM2D, Flatten, Conv1D, MaxPooling1D, \
    Concatenate, Reshape
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional
from keras import Input, Model
import tensorflow as tf
import Utility as ut

def create_uncompiled_model_bidirectional_seq2seq_mine(n_timestamps, n_future, n_features, layer_units, activation):
    tmp_model = tf.keras.models.Sequential()
    tmp_model.add(Bidirectional(LSTM(layer_units, activation=activation), input_shape = (n_timestamps, n_features)))
    tmp_model.add(RepeatVector(n_timestamps))
    tmp_model.add(Bidirectional(LSTM(layer_units)))
    tmp_model.add(Dense(n_future, activation='linear'))
    return tmp_model

def create_uncompiled_model_bidirectional_seq4seq_time(n_timestamps, n_future, n_features, layer_units, activation, dropout = 0.0):
    tmp_model = tf.keras.models.Sequential()
    tmp_model.add(Bidirectional(LSTM(layer_units, return_sequences=True, activation=activation), input_shape=(n_timestamps, n_features)))
    tmp_model.add(Bidirectional(LSTM(layer_units)))
    tmp_model.add(RepeatVector(n_timestamps))
    tmp_model.add(Bidirectional(LSTM(layer_units, return_sequences = True)))
    tmp_model.add(Bidirectional(LSTM(layer_units, return_sequences=True, dropout=dropout)))
    tmp_model.add(TimeDistributed(Dense(n_future, activation='linear')))
    return tmp_model

def create_uncompiled_model_bidirectional_seq4seq(n_timestamps, n_future, n_features, layer_units, activation):
    tmp_model = tf.keras.models.Sequential()
    tmp_model.add(Bidirectional(LSTM(layer_units, return_sequences=True, activation=activation), input_shape=(n_timestamps, n_features)))
    tmp_model.add(Bidirectional(LSTM(layer_units)))
    tmp_model.add(RepeatVector(n_timestamps))
    tmp_model.add(Bidirectional(LSTM(layer_units, return_sequences = True)))
    tmp_model.add(Bidirectional(LSTM(layer_units, return_sequences=True)))
    tmp_model.add(Bidirectional(LSTM(n_features, return_sequences=False)))
    tmp_model.add(Dense(n_future, activation='linear'))
    return tmp_model

def create_uncompiled_model_bidirectional_seq2seq(n_timestamps, n_future, n_features, layer_units, activation):
    tmp_model = tf.keras.models.Sequential()
    tmp_model.add(Bidirectional(LSTM(layer_units, activation=activation), input_shape = (n_timestamps, n_features)))
    tmp_model.add(RepeatVector(n_timestamps))
    tmp_model.add(Bidirectional(LSTM(layer_units, return_sequences=True)))
    tmp_model.add(Bidirectional(LSTM(n_features, return_sequences=False)))
    tmp_model.add(Dense(n_future, activation='linear'))
    return tmp_model

def create_uncompiled_model_bidirectional_seq2seq_time(n_timestamps, n_future, n_features, layer_units, activation):
    tmp_model = tf.keras.models.Sequential()
    tmp_model.add(Bidirectional(LSTM(layer_units, activation=activation), input_shape = (n_timestamps, n_features)))
    tmp_model.add(RepeatVector(n_timestamps))
    tmp_model.add(Bidirectional(LSTM(layer_units, return_sequences=True)))
    tmp_model.add(TimeDistributed(Dense(n_future, activation='linear')))
    return tmp_model

def create_uncompiled_model__E2D2(n_timestamps, n_future, n_features, layer_units, activation):
    encoder_inputs = tf.keras.layers.Input(shape=(n_timestamps, n_features))

    encoder_l1 = tf.keras.layers.LSTM(layer_units, return_sequences=True, return_state=True)
    encoder_outputs1 = encoder_l1(encoder_inputs)
    encoder_states1 = encoder_outputs1[1:]

    encoder_l2 = tf.keras.layers.LSTM(layer_units, return_state=True)
    encoder_outputs2 = encoder_l2(encoder_outputs1[0])
    encoder_states2 = encoder_outputs2[1:]
    #
    decoder_inputs = tf.keras.layers.RepeatVector(n_timestamps)(encoder_outputs2[0])#X_train.shape[1]
    #
    decoder_l1 = tf.keras.layers.LSTM(layer_units, return_sequences=True)(decoder_inputs, initial_state=encoder_states1)
    decoder_l2 = tf.keras.layers.LSTM(layer_units, return_sequences=True)(decoder_l1, initial_state=encoder_states2)
    decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_future))(decoder_l2)
    #
    tmp_model = tf.keras.models.Model(encoder_inputs, decoder_outputs2)
    return tmp_model

def create_uncompiled_model_E1D1(n_timestamps, n_future, n_features, layer_units, activation):
    encoder_inputs = tf.keras.layers.Input(shape=(n_timestamps, n_features))
    encoder_l1 = tf.keras.layers.LSTM(layer_units, return_state=True)
    encoder_outputs1 = encoder_l1(encoder_inputs)
    encoder_states1 = encoder_outputs1[1:]
    #
    decoder_inputs = tf.keras.layers.RepeatVector(n_timestamps)(encoder_outputs1[0])
    #
    decoder_l1 = tf.keras.layers.LSTM(layer_units, return_sequences=True)(decoder_inputs, initial_state=encoder_states1)
    decoder_outputs1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_future, activation='linear'))(decoder_l1)
    #
    tmp_model = tf.keras.models.Model(encoder_inputs, decoder_outputs1)
    return tmp_model

def create_uncompiled_model_ReturnState(n_timestamps, n_future, n_features, layer_units, activation):
    input = tf.keras.layers.Input(shape=(n_timestamps, n_features))

    lstm1 = LSTM(layer_units, return_state=True)
    LSTM_output, state_h, state_c = lstm1(input)
    states = [state_h, state_c]

    repeat = RepeatVector(n_timestamps) #1
    LSTM_output = repeat(LSTM_output)

    lstm2 = LSTM(layer_units, return_sequences=True)
    all_state_h = lstm2(LSTM_output, initial_state=states)

    # dense = TimeDistributed(Dense(y_train.shape[1], activation='linear'))
    # output = dense(all_state_h)

    lstm3 = LSTM(n_timestamps, return_sequences=False)
    all_state_d = lstm3(all_state_h)

    dense = Dense(n_timestamps, activation='linear')
    output = dense(all_state_d)

    tmp_model = Model(input, output, name='model_LSTM_return_state')
    return tmp_model

def create_uncompiled_model_bidirectional_stacked(n_timestamps, n_future, n_features, layer_units, activation):
    tmp_model = tf.keras.models.Sequential()
    tmp_model.add(Bidirectional(LSTM(layer_units, return_sequences=True, activation=activation), input_shape=(n_timestamps, n_features)))
    tmp_model.add(Bidirectional(LSTM(int(layer_units/3*2))))
    tmp_model.add(Dense(n_future, activation='linear'))
    return tmp_model

def create_uncompiled_model_bidirectional(n_timestamps, n_future, n_features, layer_units, activation):
    tmp_model = tf.keras.models.Sequential()
    tmp_model.add(Bidirectional(LSTM(layer_units, activation=activation), input_shape=(n_timestamps, n_features)))
    tmp_model.add(Dense(n_future, activation='linear'))
    return tmp_model

def create_uncompiled_model_ReturnState_ReturnSeq(n_timestamps, n_future, n_features, layer_units, activation):
    input = tf.keras.layers.Input(shape=(n_timestamps, n_features))

    lstm1 = LSTM(layer_units, return_sequences=True, return_state=True, activation=activation)
    all_state_h, state_h, state_c = lstm1(input)
    states = [state_h, state_c]

    lstm2 = LSTM(layer_units, return_sequences=True, activation=activation)
    all_state_h = lstm2(all_state_h, initial_state=states)

    # dense = TimeDistributed(Dense(n_future, activation='linear'))
    # output = dense(all_state_h)

    lstm3 = LSTM(n_features+1, return_sequences=False)
    all_state_d = lstm3(all_state_h)

    dense = Dense(n_future, activation='linear')
    output = dense(all_state_d)

    tmp_model = Model(input, output, name='model_LSTM_return_state')
    return tmp_model

def create_uncompiled_model_ReturnState_ReturnSeq_2Layers(n_timestamps, n_future, n_features, layer_units, activation, l1=0.0, l2=0.0):
    input = tf.keras.layers.Input(shape=(n_timestamps, n_features))

    lstm1 = LSTM(layer_units, return_sequences=True, return_state=True,
                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                 activation=activation)
    all_state_h, state_h, state_c = lstm1(input)
    states = [state_h, state_c]

    lstm2 = LSTM(layer_units, return_sequences=False, activation=activation)
    all_state_h = lstm2(all_state_h, initial_state=states)

    dense = Dense(n_future, activation='linear')
    output = dense(all_state_h)

    tmp_model = Model(input, output, name='model_LSTM_return_state')
    return tmp_model

def create_uncompiled_model_Gillo(n_timestamps, n_future, n_features, layer_units, activation, dropout = 0.0):
    input = tf.keras.layers.Input(shape=(n_timestamps, n_features))

    lstm1 = LSTM(layer_units, return_sequences=True, return_state=True,activation=activation)
    all_state_h, state_h, state_c = lstm1(input)
    states = [state_h, state_c]

    lstm2 = LSTM(layer_units, return_sequences=False, activation=activation)
    all_state_h = lstm2(all_state_h, initial_state=states)

    drop_layer = Dropout(dropout)
    state_drop = drop_layer(all_state_h)

    dense_last = Dense(n_features+1, activation='linear')
    output_dense = dense_last(state_drop)

    dense = Dense(n_future, activation='linear')
    output = dense(output_dense)

    tmp_model = Model(input, output, name='model_LSTM_return_state_dense')
    return tmp_model

def create_uncompiled_model_StackedMulti(n_timestamps, n_future, n_features, layer_units, activation):
    tmp_model = tf.keras.models.Sequential()
    tmp_model.add(LSTM(layer_units, activation=activation, return_sequences=True, input_shape=(n_timestamps, n_features)))
    tmp_model.add(LSTM(256, activation=activation, return_sequences=True))
    tmp_model.add(LSTM(128, activation=activation, return_sequences=True))
    tmp_model.add(LSTM(128, activation=activation))
    tmp_model.add(Dense(64, activation=activation))
    tmp_model.add(Dense(n_future, activation='linear'))
    return tmp_model

def create_uncompiled_model_Stacked(n_timestamps, n_future, n_features, layer_units, activation, l1=0.0, l2=0.0):
    tmp_model = tf.keras.models.Sequential()
    tmp_model.add(LSTM(layer_units, return_sequences=True,
                       activation=activation,
                       kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                       stateful=False,
                       input_shape=(n_timestamps, n_features)))
    tmp_model.add(LSTM(int(layer_units*3/2)))  # Layer 2
    tmp_model.add(Dense(n_future, activation='linear'))
    return tmp_model

def create_uncompiled_model_vanilla(n_timestamps, n_future, n_features, layer_units, activation, l1=0.0, l2=0.0):
    tmp_model = tf.keras.models.Sequential()
    tmp_model.add(LSTM(layer_units, return_sequences=False,
                       activation=activation,
                       kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                       stateful=False,
                       input_shape=(n_timestamps, n_features)))
    tmp_model.add(Dense(n_future,  activation='linear'))
    return tmp_model

def create_uncompiled_model_tiny(n_timestamps, n_future, n_features, activation, l1=0.0, l2=0.0):
    tmp_model = tf.keras.models.Sequential()
    tmp_model.add(LSTM(units=n_features+1,
                       return_sequences=False,
                       activation=activation,
                       recurrent_dropout=0.0,
                       kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                       input_shape=(n_timestamps, n_features)))
    tmp_model.add(Dense(n_future))
    return tmp_model

def create_uncompiled_gru(n_timestamps, n_future, n_features, layer_units, activation):
    tmp_model = tf.keras.models.Sequential()  # Model
    tmp_model.add(Input(shape=(n_timestamps, n_features)))  # Input Layer - need to speicfy the shape of inputs
    tmp_model.add(Bidirectional(GRU(units=layer_units, activation=activation, recurrent_activation='sigmoid', stateful=False)))  # Encoder Layer
    tmp_model.add(RepeatVector(n_timestamps))  # Repeat Vector
    tmp_model.add(Bidirectional(GRU(units=layer_units, activation=activation, recurrent_activation='sigmoid', stateful=False, return_sequences=False)))  # Decoder Layer
    tmp_model.add(Dense(units=1, activation='linear'))  # Output Layer, Linear(x) = x
    # tmp_model.add(TimeDistributed(Dense(units=1, activation='linear'), name='Output-Layer'))  # Output Layer, Linear(x) = x
    return tmp_model

def create_uncompiled_gru_simple(n_timestamps, n_future, n_features, layer_units, activation):
    tmp_model = tf.keras.models.Sequential()  # Model
    tmp_model.add(Input(shape=(n_timestamps, n_features)))  # Input Layer - need to speicfy the shape of inputs
    tmp_model.add(GRU(units=layer_units, activation=activation, recurrent_activation='sigmoid', stateful=False, return_sequences=True))
    tmp_model.add(GRU(units=layer_units, activation=activation, recurrent_activation='sigmoid', stateful=False, return_sequences=False))
    tmp_model.add(Dense(units=n_future, activation='linear'))  # Output Layer, Linear(x) = x
    return tmp_model

def create_uncompiled_conv(n_timestamps, n_future, n_features, layer_units, activation):
    tmp_model = tf.keras.models.Sequential()  # Model
    tmp_model.add(ConvLSTM2D(filters=64, kernel_size=(1, 2), activation='relu', input_shape=(n_timestamps, 1, n_future, n_features)))
    tmp_model.add(Flatten())
    tmp_model.add(Dense(1))
    return tmp_model

def create_CNN_LSTM(n_timestamps, n_future, n_features, layer_units, activation):
    tmp_model = tf.keras.models.Sequential()  # Model
    tmp_model.add(Conv1D(filters=64, kernel_size=9, activation=activation, input_shape = (n_timestamps, n_features)))
    tmp_model.add(Conv1D(filters=64, kernel_size=11, activation=activation))
    tmp_model.add(MaxPooling1D(pool_size=2))
    tmp_model.add(Flatten())
    tmp_model.add(RepeatVector(n_future))
    tmp_model.add(LSTM(200, activation=activation, return_sequences = False))
    tmp_model.add(Dense(units=n_features+1, activation='linear'))  # Output Layer, Linear(x) = x
    tmp_model.add(Dense(units=n_future, activation='linear'))  # Output Layer, Linear(x) = x
    return tmp_model

def create_MultiHead(n_timestamps, n_future, n_features, layer_units, activation):
    input_layer = Input(shape=(n_timestamps, n_features))
    head_list = []
    for i in range(0, n_features):
        conv_layer_head = Conv1D(filters=4, kernel_size=7, activation=activation)(input_layer)
        conv_layer_head_2 = Conv1D(filters=6, kernel_size=11, activation=activation)(conv_layer_head)
        conv_layer_flatten = Flatten()(conv_layer_head_2)
        head_list.append(conv_layer_flatten)

    concat_cnn = Concatenate(axis=1)(head_list)
    reshape = Reshape((head_list[0].shape[1], n_features))(concat_cnn)
    lstm = LSTM(100, activation=activation)(reshape)
    repeat = RepeatVector(n_future)(lstm)
    lstm_2 = LSTM(100, activation=activation, return_sequences = True)(repeat)
    dropout = Dropout(0.1)(lstm_2)
    dense = Dense(n_future, activation='linear')(dropout)
    tmp_model = Model(inputs=input_layer, outputs=dense)
    return tmp_model

def compile_model(model, optimizer, loss):
    model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=[tf.keras.metrics.RootMeanSquaredError(),
                               tf.keras.metrics.MeanAbsoluteError(),
                               tf.keras.metrics.MeanSquaredError()])
    return model
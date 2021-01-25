import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf

from model.Decoder import Decoder
from model.Encoder import Encoder
from model.Surrogate import Surrogate

from test_pop import gen_pop

def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}
    ax.set_xticklabels([''] + sentence.tolist(), fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence.tolist(), fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def cut_seq(seq, end_token=2):
    ind = (seq == end_token).argmax(1)
    res = [np.pad(d[:i + 1], (0, max_size - i - 1)) for d, i in
           zip(seq, ind)]
    return res


batch_size = 1
max_size = 40
vocab_inp_size = 15
vocab_tar_size = 15
embedding_dim = 64
units = 256
hidden_size = 256
alpha = 0.8
epochs = 1
epoch_decay = 1
min_epochs = 10
verbose = True

enc = Encoder(vocab_inp_size, embedding_dim, units, batch_size)
dec = Decoder(vocab_inp_size, vocab_tar_size, embedding_dim, units, batch_size)
enc.eval()
dec.eval()
surrogate = Surrogate(hidden_size)

train_steps = 0
save_path = "2021-01-24_13:29:43.553851"
enc.load_weights("model/weights/{}/encoder/enc_{}".format(save_path, train_steps))
dec.load_weights("model/weights/{}/decoder/dec_{}".format(save_path, train_steps))
surrogate.load_weights(
    "model/weights/{}/surrogate/surrogate_{}".format(save_path, train_steps))
print("Weights loaded")

def evaluate(seq, fit):
    inp = tf.constant([seq + [0] * (max_size - len(seq))],
                                      dtype=tf.int64)
    # example_target_batch = tf.random.uniform(shape=(batch_size, max_size), minval=0,
    #                                          maxval=15, dtype=tf.int64)
    latent = enc(inp)
    print("Latent shape:", latent.shape)
    surr_out = surrogate(latent)

    dec_input = tf.expand_dims([1] * len(inp), 1)
    predictions = dec(latent, max_size-2)

    predicted_tokens = tf.argmax(predictions, axis=2, output_type=tf.dtypes.int32)
    stop_tokens = tf.expand_dims([2] * len(inp), 1)
    child = tf.concat([dec_input,
                       predicted_tokens,
                       stop_tokens], axis=1)
    child = child.numpy()[0]
    # child = np.array(cut_seq(child, end_token=2))
    print("Input:", inp.numpy()[0])
    print("Output:", child)
    print("Fitness:", surr_out.numpy()[0][0], "True fit:", fit)

    return surr_out.numpy()[0][0]


pop, fit = gen_pop(max_size)
fitness = []
t_fit = []
max_pop = 1
# seq = np.random.randint(1,11, size=(100,max_size))
for p, f in zip(pop[:max_pop], fit[:max_pop]):
# for p in seq:
    print(p, f)
    # p = p.tolist()
    fit = evaluate(p, f)
    # if f>0.5:
    #     t_fit = f
    #     fitness.append(fit)
# print("Fitnes MAE", np.mean(np.abs(np.subtract(t_fit, fitness))))
# print("Fitnes MSE", np.mean(np.square(np.subtract(t_fit, fitness))))

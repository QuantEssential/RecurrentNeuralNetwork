


from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer

from pybrain.datasets import SequentialDataSet
from itertools import cycle

from pybrain.supervised import RPropMinusTrainer
from sys import stdout


def run_network(signal, signal_2):

    ds = SequentialDataSet(1, 1)
    for sample, next_sample in zip(signal_2[-333:-66], cycle(signal_2[-300:-33])):
        ds.addSample(sample, next_sample)


    net = buildNetwork(1, 10, 1,
                       hiddenclass=LSTMLayer, outputbias=False, recurrent=True)

    trainer = RPropMinusTrainer(net, dataset=ds)
    train_errors = []  # save errors for plotting later
    EPOCHS_PER_CYCLE = 5
    CYCLES = 33
    EPOCHS = EPOCHS_PER_CYCLE * CYCLES
    for i in xrange(CYCLES):
        trainer.trainEpochs(EPOCHS_PER_CYCLE)
        train_errors.append(trainer.testOnData())
        epoch = (i + 1) * EPOCHS_PER_CYCLE
        print("\r epoch {}/{}".format(epoch, EPOCHS))
        stdout.flush()


    ds = SequentialDataSet(1, 1)
    for sample, next_sample in zip(signal_2[-333:-33], cycle(signal_2[-300:])):
        ds.addSample(sample, next_sample)

    last_prediction = 0


    predictions = []
    count=0



    for sample, target in ds.getSequenceIterator(0):
        print("               sample = %4.1f" % sample)
        print("predicted next sample = %4.1f" % net.activate(sample))
        print("   actual next sample = %4.1f" % target)
        last_prediction = float(net.activate(sample))
        predictions.append(last_prediction)

        count+=1






    ds = SequentialDataSet(1, 1)
    for sample, next_sample in zip(signal[-333:-66], cycle(signal[-300:-33])):
        ds.addSample(sample, next_sample)

    net = buildNetwork(1, 10, 1,
                       hiddenclass=LSTMLayer, outputbias=False, recurrent=True)

    trainer = RPropMinusTrainer(net, dataset=ds)
    train_errors = []  # save errors for plotting later
    EPOCHS_PER_CYCLE = 5
    CYCLES = 33
    EPOCHS = EPOCHS_PER_CYCLE * CYCLES
    for i in xrange(CYCLES):
        trainer.trainEpochs(EPOCHS_PER_CYCLE)
        train_errors.append(trainer.testOnData())
        epoch = (i + 1) * EPOCHS_PER_CYCLE
        print("\r epoch {}/{}".format(epoch, EPOCHS))
        stdout.flush()



    ds = SequentialDataSet(1, 1)
    for sample, next_sample in zip(signal[-333:-33], cycle(signal[-300:])):
        ds.addSample(sample, next_sample)


    last_prediction = 0
    predictions = []
    count=0

    for sample, target in ds.getSequenceIterator(0):
        print("               sample = %4.1f" % sample)
        print("predicted next sample = %4.1f" % net.activate(sample))
        print("   actual next sample = %4.1f" % target)
        last_prediction = float(net.activate(sample))
        predictions.append(last_prediction)
        count+=1

    for i in range(0, 10):
        print("predicted next sample = %4.1f" % net.activate(last_prediction))
        last_prediction = float(net.activate(last_prediction))



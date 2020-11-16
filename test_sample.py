# from crepe.cli import run
import core
if __name__ == '__main__':
    # import os
    # filename = [os.path.join(os.path.dirname(__file__), 'tests/sample_data/gap_2.wav')]
    # run(filename, viterbi=False)
    core.train(task='raga', tradition='hindustani')
    # core.train(task='tonic', tradition='hindustani')
    # core.test(task='raga', tradition='hindustani')
    # core.train_pitch(tradition='hindustani')
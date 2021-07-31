# import transformers.midi_processor.processor as sequence

# RANGE_NOTE_ON = 128
# RANGE_NOTE_OFF = 128
# RANGE_VEL = 32
# RANGE_TIME_SHIFT = 100

# START_IDX = {
#     'note_on': 0,
#     'note_off': RANGE_NOTE_ON,
#     'time_shift': RANGE_NOTE_ON + RANGE_NOTE_OFF,
#     'velocity': RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT
# }


# max_seq = 2048
max_seq=2048
l_r = 0.001
embedding_dim = 256
num_attention_layer = 6
batch_size = 1
loss_type = 'categorical_crossentropy'
# event_dim = RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT + RANGE_VEL
event_dim = 60
# pad_token = 60
pad_token = event_dim
token_sos = event_dim + 1
token_eos = event_dim + 2
vocab_size = event_dim + 3

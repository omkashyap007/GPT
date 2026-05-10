modes = {
    1: "pretraining",
    2: "qa_training"
}
mode = 1


batch_size = 64
block_size = 256
eval_interval = 500
eval_iters = 200
learning_rate = 1e-4
training_iteration = 1 # times the model has been updated and values has been tweaked
base_training_file_name = "trained_model/plain_english"
qa_training_file_name = "trained_model/qa_text"
train_file_name = base_training_file_name
text_training_steps = 50000
qa_training_steps = 30000
checkpoint_start = 0

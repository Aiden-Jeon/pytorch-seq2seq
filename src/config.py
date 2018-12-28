cfg ={
    # “Teacher forcing” is the concept of using the real target outputs as each next input,
    # instead of using the decoder’s guess as the next input.
    "teacher_forcing_ratio": 0.5,
    "hidden_size": 50,
    "max_length": 50,
    "n_epochs": 10,
    "batch_size": 1,
    "save_steps": 5,
    "learning_rate": 0.1
}
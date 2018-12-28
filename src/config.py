cfg ={
    # “Teacher forcing” is the concept of using the real target outputs as each next input,
    # instead of using the decoder’s guess as the next input.
    "teacher_forcing_ratio": 0.5,
    "hidden_size": 256,
    "max_length": 50,
    "n_epochs":10
}
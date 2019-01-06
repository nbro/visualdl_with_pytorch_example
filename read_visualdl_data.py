from visualdl import LogReader

log_reader = LogReader("./log")

print("Data associated with the train loss:\n")
with log_reader.mode("train") as logger:
    text_reader = logger.scalar("scalars/train_loss")
    print("Train loss =", text_reader.records())
    print("Ids = ", text_reader.ids())
    print("Timestamps =", text_reader.timestamps())

print("\nData associated with the test loss:\n")

with log_reader.mode("test") as logger:
    text_reader = logger.scalar("scalars/test_loss")
    print("Test losses =", text_reader.records())
    print("Ids = ", text_reader.ids())
    print("Timestamps =", text_reader.timestamps())

print("\nData associated with the test accuracy:\n")

with log_reader.mode("test") as logger:
    text_reader = logger.scalar("scalars/test_accuracy")
    print("Test accuracy =", text_reader.records())
    print("Ids = ", text_reader.ids())
    print("Timestamps =", text_reader.timestamps())

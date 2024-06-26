from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import DataSet
import time
import os.path

def train(data_type, seq_length, model, saved_model=None,
          class_limit=None,
          load_to_memory=False, batch_size=32, nb_epoch=5):
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('checkpoints', model + '-' + data_type + \
            '.{epoch:03d}-{val_loss:.3f}.keras'),
        verbose=1,
        save_best_only=True)

    tb = TensorBoard(log_dir=os.path.join('logs', model))

    early_stopper = EarlyStopping(patience=5)

    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('logs', model + '-' + 'training-' + \
        str(timestamp) + '.log'))

    data = DataSet(
        seq_length=seq_length,
        class_limit=class_limit
    )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.7) // batch_size


    X, y = data.get_all_sequences_in_memory('train', data_type)
    X_test, y_test = data.get_all_sequences_in_memory('test', data_type)


    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)


    rm.model.fit(
        X,
        y,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[checkpointer, tb, early_stopper, csv_logger],
        epochs=nb_epoch)
    
    # Save the final model
    final_model_path = os.path.join('checkpoints', f'{model}-{data_type}-final.keras')
    rm.model.save(final_model_path)
    print(f'Model saved to {final_model_path}')


def main():
    model = 'lstm'
    class_limit = 2
    seq_length = 40
    batch_size = 32
    nb_epoch = 100
    data_type = 'features'
    load_to_memory = True


    train(data_type, seq_length, model,
          class_limit=class_limit, load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()

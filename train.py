from model import *
from config import *

if os.path.exists('modello7.h5'):
    model = tf.keras.models.load_model('modello7.h5')
else:
    train_images_files = glob.glob(train_images_path + '/*.npy')
    train_masks_files = glob.glob(train_masks_path + '/*.npy')

    encoder_filters = [16, 32, 64, 128]
    decoder_filters = encoder_filters[::-1]
    kernel = 3
    input_shape = (256, 256, 10)
    num_classes = 2
    epoch = 10
    batch_size = 10
    num_parts = 10
    part_size = len(train_images_files) // num_parts

    model = create_unet(input_shape, num_classes, kernel, encoder_filters, decoder_filters)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    for batch_idx in range(num_parts):
        start_idx = batch_idx * part_size
        end_idx = (batch_idx + 1) * part_size

        train_images_batch = np.array([np.load(file) for file in train_images_files[start_idx:end_idx]])
        train_masks_batch = np.array([np.load(file) for file in train_masks_files[start_idx:end_idx]])

        new_size = (256, 256)
        train_images_resized = tf.image.resize(train_images_batch, new_size)
        train_masks_resized = np.zeros((train_masks_batch.shape[0], *new_size))
        for i in range(train_masks_batch.shape[0]):
            mask = Image.fromarray(train_masks_batch[i])
            mask_enlarged = mask.resize(new_size)
            train_masks_resized[i] = np.array(mask_enlarged)

        model.fit(train_images_resized, train_masks_resized, epochs=epoch, batch_size=batch_size)
        print("Parte ", batch_idx, " finita")

    model.save('modello7.h5')

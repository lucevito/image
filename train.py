from model import *
from config import *

if os.path.exists('modello.h5'):
    model = tf.keras.models.load_model('modello.h5')
else:
    model = create_unet(input_shape, num_classes, kernel, encoder_filters, decoder_filters)
    model.compile(optimizer='adam', loss=weighted_loss, metrics=['accuracy'])

num_parts = 20
epoch = 400
batch_size = 10
ripetizioni = 1
for k in range(ripetizioni):
    for batch_idx in range(num_parts):
        start_idx = batch_idx * len(train_images_files) // num_parts
        end_idx = (batch_idx + 1) * len(train_images_files) // num_parts

        train_images_batch = np.array([np.load(file) for file in train_images_files[start_idx:end_idx]])
        train_masks_batch = np.array([np.load(file) for file in train_masks_files[start_idx:end_idx]])
        train_images_batch = train_images_batch[:, :, :, :canaleI:canaleF]

        train_images_resized = tf.image.resize(train_images_batch, new_size)
        train_masks_resized = np.zeros((train_masks_batch.shape[0], *new_size))
        for i in range(train_masks_batch.shape[0]):
            mask = Image.fromarray(train_masks_batch[i])
            mask_enlarged = mask.resize(new_size)
            train_masks_resized[i] = np.array(mask_enlarged)

        model.fit(train_images_resized, train_masks_resized, epochs=epoch, batch_size=batch_size)
        print("Parte : ", batch_idx + 1, " finita su :", num_parts)
        print("==============================================")

    model.save('modello.h5')



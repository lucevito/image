from config import *
from model import *

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support

modello = 'modello.h5'
if os.path.exists(modello):
    with tf.keras.utils.custom_object_scope({'weighted_loss': weighted_loss}):
        model = tf.keras.models.load_model('modello.h5')

    test_images_path = 'Immagini_satellitari/Test/images'
    test_masks_path = 'Immagini_satellitari/Test/masks'
    test_images_files = glob.glob(test_images_path + '/*.npy')
    test_masks_files = glob.glob(test_masks_path + '/*.npy')
    num_parts = 2

    for batch_idx in range(num_parts):
        start_idx = batch_idx * len(test_images_files) // num_parts
        end_idx = (batch_idx + 1) * len(test_images_files) // num_parts

        test_images_batch = np.array([np.load(file) for file in test_images_files[start_idx:end_idx]])
        test_masks_batch = np.array([np.load(file) for file in test_masks_files[start_idx:end_idx]])
        test_images_batch = test_images_batch[:, :, :, canali_selezionati]

        new_size = (128, 128)
        test_images_batch = tf.image.resize(test_images_batch, new_size)
        resized_dataset = []
        for image in test_masks_batch:
            resized_image = resize_image(image, new_size)
            resized_dataset.append(resized_image)
        test_masks_batch = np.array(resized_dataset)

        predicted_masks = model.predict(test_images_batch)
        predicted_masks = np.argmax(predicted_masks, axis=-1)
        return_size = (32, 32)
        resized_dataset = []
        for image in predicted_masks:
            resized_image = resize_image(image, new_size)
            resized_dataset.append(resized_image)
        predicted_masks = np.array(resized_dataset)

        true_masks = test_masks_batch.flatten()
        pred_masks = predicted_masks.flatten()

        print("============================================================================")
        print("Prediction : \n")
        print("Evaluation Metrics:")
        print("Precision:", precision_score(true_masks, pred_masks, zero_division=0))
        print("Recall:", recall_score(true_masks, pred_masks, zero_division=0))
        print("F1-score:", f1_score(true_masks, pred_masks, zero_division=0))
        print("Mean IoU:", jaccard_score(true_masks, pred_masks, average='macro'))
        print("Confusion Matrix:\n", confusion_matrix(true_masks, pred_masks))

        metrics = precision_recall_fscore_support(true_masks.ravel(), pred_masks.ravel(), zero_division=0, average=None)
        precision_attacchi = metrics[1][1]
        precision_normali = metrics[1][0]
        f1_attacchi = metrics[2][1]
        f1_normali = metrics[2][0]
        macro_f1 = metrics[2].mean()
        oa = accuracy_score(true_masks, pred_masks)
        print("precision_attacchi: ", f1_attacchi)
        print("precision_normali: ", f1_normali)
        print("F1 attacchi:", f1_attacchi)
        print("F1 normali:", f1_normali)
        print("Macro F1:", macro_f1)
        print("OA:", oa)

        visualize_pixel_plots(test_images_batch, test_masks_batch, predicted_masks)
        print("Parte : ", batch_idx + 1, " finita su :", num_parts)
        print("==============================================")

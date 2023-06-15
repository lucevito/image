from config import *
from model import *


if os.path.exists('modello.h5'):
    with tf.keras.utils.custom_object_scope({'weighted_loss': weighted_loss}):
        model = tf.keras.models.load_model('modello.h5')
    test_images = np.array([np.load(file) for file in glob.glob(test_images_path + '/*.npy')])
    test_masks = np.array([np.load(file) for file in glob.glob(test_masks_path + '/*.npy')])

    test_images = tf.image.resize(test_images,new_size)
    resized_dataset = []
    for image in test_masks:
        resized_image = resize_image(image,new_size)
        resized_dataset.append(resized_image)
    test_masks = np.array(resized_dataset)

    predicted_masks = model.predict(test_images)
    predicted_masks = np.argmax(predicted_masks, axis=-1)


    return_size = (32, 32)
    resized_dataset = []
    for image in predicted_masks:
        resized_image = resize_image(image,new_size)
        resized_dataset.append(resized_image)
    predicted_masks = np.array(resized_dataset)

    true_masks = test_masks.flatten()
    pred_masks = predicted_masks.flatten()

    print("============================================================================")
    print("Prediction : \n")
    print("Evaluation Metrics:")
    print("Precision:", precision_score(true_masks, pred_masks, zero_division=0))
    print("Recall:", recall_score(true_masks, pred_masks, zero_division=0))
    print("F1-score:", f1_score(true_masks, pred_masks, zero_division=0))
    print("Mean IoU:", jaccard_score(true_masks, pred_masks, average='macro'))
    print("Confusion Matrix:\n", confusion_matrix(true_masks, pred_masks))
    visualize_pixel_plots(test_images, test_masks, predicted_masks)

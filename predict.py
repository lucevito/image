from config import *
from model import *

if os.path.exists('modello.h5'):
    with tf.keras.utils.custom_object_scope({'weighted_loss': weighted_loss}):
        model = tf.keras.models.load_model('modello.h5')
    test_images = np.array([np.load(file) for file in glob.glob(test_images_path + '/*.npy')])
    test_masks = np.array([np.load(file) for file in glob.glob(test_masks_path + '/*.npy')])

    test_images_resized = tf.image.resize(test_images, new_size)
    test_masks_resized = np.zeros((test_masks.shape[0], *new_size))
    for i in range(test_masks.shape[0]):
        mask = Image.fromarray(test_masks[i])
        mask_enlarged = mask.resize(new_size)
        test_masks_resized[i] = np.array(mask_enlarged)

    predicted_masks = model.predict(test_images_resized)

    new_size = (32, 32)
    predicted_masks = tf.image.resize(predicted_masks, new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    predicted_masks = np.argmax(predicted_masks, axis=-1)

    true_masks = test_masks.flatten()
    pred_masks = predicted_masks.flatten()

    print("============================================================================")
    print("Prediction : \n")
    print("Evaluation Metrics:")
    print("Precision:", precision_score(true_masks, pred_masks, zero_division=0))
    print("Recall:", recall_score(true_masks, pred_masks))
    print("F1-score:", f1_score(true_masks, pred_masks))
    print("Mean IoU:", jaccard_score(true_masks, pred_masks, average='macro'))
    print("Confusion Matrix:\n", confusion_matrix(true_masks, pred_masks))
    visualize_pixel_plots(test_images, test_masks, predicted_masks)

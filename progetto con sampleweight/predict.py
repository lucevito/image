from config import *
from model import *
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support

if os.path.exists(modello):
    model = tf.keras.models.load_model(modello)

    # T E S T
    test_images_path = 'Immagini_satellitari/Test/images'
    test_masks_path = 'Immagini_satellitari/Test/masks'
    test_images_files = glob.glob(test_images_path + '/*.npy')
    test_masks_files = glob.glob(test_masks_path + '/*.npy')
    test_images_batch = np.array([np.load(file) for file in test_images_files])
    test_masks_batch = np.array([np.load(file) for file in test_masks_files])
    test_images_batch = test_images_batch[:, :, :, canali_selezionati]

    predicted_masks = model.predict(test_images_batch)
    predicted_masks = (predicted_masks > 0.5).astype(np.uint8)
    true_masks = test_masks_batch.flatten()
    pred_masks = predicted_masks.flatten()

    print("============================================================================")
    print("Prediction : TEST \n")
    print("Evaluation Metrics:")
    print("Precision:", precision_score(true_masks, pred_masks, zero_division=0))
    print("Recall:", recall_score(true_masks, pred_masks, zero_division=0))
    print("F1-score:", f1_score(true_masks, pred_masks, zero_division=0))
    print("Mean IoU:", jaccard_score(true_masks, pred_masks, average='macro'))
    print("Confusion Matrix:\n", confusion_matrix(true_masks, pred_masks))
    metrics = precision_recall_fscore_support(true_masks.ravel(), pred_masks.ravel(), zero_division=0 ,average=None)
    print("precision_attacchi: ", metrics[1][1])
    print("precision_normali: ", metrics[1][0])
    print("F1 attacchi:", metrics[2][1])
    print("F1 normali:", metrics[2][0])
    print("Macro F1:", metrics[2].mean())
    print("OA:", accuracy_score(true_masks, pred_masks))
    #for i in range(len(test_images_batch)):
    #  visualize_pixel_plots(test_images_batch[i], test_masks_batch[i], predicted_masks[i], 'output/test')

    # T R A I N
    train_images_path = 'Immagini_satellitari/Train/images'
    train_masks_path = 'Immagini_satellitari/Train/masks'
    train_images_files = glob.glob(train_images_path + '/*.npy')
    train_masks_files = glob.glob(train_masks_path + '/*.npy')
    train_images_batch = np.array([np.load(file) for file in train_images_files])
    train_masks_batch = np.array([np.load(file) for file in train_masks_files])
    train_images_batch = train_images_batch[:, :, :, canali_selezionati]

    predicted_masks = model.predict(train_images_batch)
    predicted_masks = (predicted_masks > 0.5).astype(np.uint8)
    true_masks = train_masks_batch.flatten()
    pred_masks = predicted_masks.flatten()

    print("============================================================================")
    print("Prediction : TRAIN \n")
    print("Evaluation Metrics:")
    print("Precision:", precision_score(true_masks, pred_masks, zero_division=0))
    print("Recall:", recall_score(true_masks, pred_masks, zero_division=0))
    print("F1-score:", f1_score(true_masks, pred_masks, zero_division=0))
    print("Mean IoU:", jaccard_score(true_masks, pred_masks, average='macro'))
    print("Confusion Matrix:\n", confusion_matrix(true_masks, pred_masks))
    metrics = precision_recall_fscore_support(true_masks.ravel(), pred_masks.ravel(), zero_division=0 ,average=None)
    print("precision_attacchi: ", metrics[1][1])
    print("precision_normali: ", metrics[1][0])
    print("F1 attacchi:", metrics[2][1])
    print("F1 normali:", metrics[2][0])
    print("Macro F1:", metrics[2].mean())
    print("OA:", accuracy_score(true_masks, pred_masks))
    #for i in range(len(train_images_batch)):
    #  visualize_pixel_plots(train_images_batch[i], train_masks_batch[i], predicted_masks[i], 'output/train')

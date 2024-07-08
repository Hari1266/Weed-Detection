from concurrent.futures import ProcessPoolExecutor
from symbol import tfpdef
import cv2
import numpy as np
from tqdm import tqdm

# Load models once
model_path = 'cnn.h5'
svm_model_path = 'svm.pkl'
model = tfpdef.keras.models.load_model(model_path)
model_without_last_two_fc = tf.keras.models.Model(model.inputs, model.layers[-5].output)

with open(svm_model_path, 'rb') as svm:
    svm_model = pickle.load(svm)

# Existing constants and functions

def process_region(rect, img):
    x, y, w, h = rect
    roi = img[y:y+h, x:x+w, :]
    resized_roi = cv2.resize(roi, (224, 224)) / 255
    feature = model_without_last_two_fc.predict(resized_roi.reshape(-1, 224, 224, 3))
    pred = svm_model.predict_proba(feature.reshape(-1, 4096))
    pred_lab = svm_model.predict(feature.reshape(-1, 4096))
    return [list(rect), np.max(pred), pred_lab]

def parallel_detection(img_path, confidence=0.9, iou_thresh=0.1):
    img = plt.imread(img_path)
    cv2.setUseOptimized(True)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    sel_rects = ss.process()[:100]

    def process_single_region(rect):
        return process_region(rect, img)

    # Parallelize the processing of regions using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_single_region, sel_rects), total=len(sel_rects)))

    final = []
    for rect, score, cls in results:
        if cls == 'crop' and np.max(score) > confidence:
            final.append([rect, np.max(score), 'crop'])
        elif cls == 'weed' and np.max(score) > confidence:
            final.append([rect, np.max(score), 'weed'])

    imOut = img.copy()
    for rect, score, cls in final:
        x, y, w, h = rect
        color = (0, 255, 0) if cls == 'crop' else (255, 0, 0)
        cv2.rectangle(imOut, (x, y), (x+w, y+h), color, 2)
        cv2.putText(imOut, f"{cls}:{round(score*100, 2)}", (x+2, y-12), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    return imOut, cls

# Example usage
if __name__ == "__main__":
    st.title("Weed Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "tif"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        pred, cls = parallel_detection(uploaded_file)

        st.subheader("Prediction: ")
        st.image(pred, caption="Prediction", use_column_width=True)
        st.markdown(f"# {cls}")

from ultralytics import YOLO

# The models are too large to check into GitHub.  To get a new model,
# run the training/football_training_yolo_v5.ipynb Jupyter Notebook.
# If training takes too long locally, consider using Google Colabs.
model = YOLO('models/best.pt')

results = model.predict('input_videos/08fd33_4.mp4', save=True)
print(results[0])
print('===================================')
for box in results[0].boxes:
    print(box)
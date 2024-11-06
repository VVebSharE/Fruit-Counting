# need to create a object counting model for fruits
- step 1: download dataset , downloaded at 
    app input: images of fruits formats: image, video, webcam, camera, folder
    app output: number of fruits in the image
- step 2: literature review checking various mthods of modeling this problem
    model input: batch of images of shape (B, 224, 224, 3)
    model output:
    - approach 0: using non-deep learning model 
    - approach 1: doing regression on the number of fruits       ----------------------------BASELINE
    - approach 2: doing classification on the number of fruits
    - approach 3: doing object detection on the number of fruits
        app
- step 3: list important models.

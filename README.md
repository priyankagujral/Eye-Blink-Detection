# Eye-Blink-Detection
Eye Blink Detection is done by using dlib and python.
After detcting the face, 68 points mapping is applied on the detected loaction of face and left eye, right eye are detected.
EAR (eye aspect ratio) is calculated in order to detect the blink.
If calculated EAR value is decreased by somethreshold vakue and then increased above the threshold value consecutively. eye blink is detected.
